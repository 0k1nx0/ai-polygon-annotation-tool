from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import cv2
import json
import os
import sqlite3
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from datetime import datetime
from typing import List, Optional
import threading
import requests

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/annotations", exist_ok=True)
os.makedirs("feedback", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "0k1nx0/polyannot-feedback")
DB_PATH    = "feedback/feedback.db"


def backup_to_hf():
    """
    Runs in a background thread after every feedback save.
    Exports feedback DB to JSON and pushes to HuggingFace Dataset repo.
    Silently fails if token is not set (local dev mode).
    """
    if not HF_TOKEN:
        return  # Skip backup in local dev mode

    try:
        # Export SQLite → JSON
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT * FROM feedback ORDER BY id")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        con.close()

        records = []
        for row in rows:
            rec = dict(zip(cols, row))
            if rec.get("ai_points"):
                rec["ai_points"] = json.loads(rec["ai_points"])
            if rec.get("corrected_points"):
                rec["corrected_points"] = json.loads(rec["corrected_points"])
            records.append(rec)

        content = json.dumps(records, indent=4)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        # Push to HuggingFace Dataset repo via API
        # Saves as feedback/feedback_latest.json (always overwrite)
        # AND feedback/history/feedback_TIMESTAMP.json (archive copy)
        for path in [
            "feedback/feedback_latest.json",
            f"feedback/history/feedback_{timestamp}.json"
        ]:
            url = f"https://huggingface.co/api/datasets/{HF_DATASET}/upload/{path}"
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json"
                },
                data=content.encode("utf-8")
            )
            if response.status_code not in (200, 201):
                print(f"[HF Backup] Failed to upload {path}: {response.text}")
            else:
                print(f"[HF Backup] Uploaded {path} successfully")

    except Exception as e:
        print(f"[HF Backup] Error: {e}")


# ────────────────────────────────────────────────
# DATABASE SETUP  (SQLite)
# ────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # feedback table — one row per annotation the user rated
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name      TEXT NOT NULL,
            annotation_id   INTEGER NOT NULL,
            class_name      TEXT NOT NULL,
            mode            TEXT NOT NULL,
            vote            TEXT NOT NULL,        -- 'up' or 'down'
            ai_points       TEXT,                 -- JSON: original AI polygon
            corrected_points TEXT,               -- JSON: user-corrected polygon (NULL if thumbs up)
            confidence      REAL,
            created_at      TEXT NOT NULL
        )
    """)

    # stats table — aggregate accuracy per class
    cur.execute("""
        CREATE TABLE IF NOT EXISTS class_stats (
            class_name  TEXT PRIMARY KEY,
            total       INTEGER DEFAULT 0,
            correct     INTEGER DEFAULT 0,
            incorrect   INTEGER DEFAULT 0
        )
    """)

    con.commit()
    con.close()

init_db()


# ────────────────────────────────────────────────
# LOAD MODELS
# ────────────────────────────────────────────────

yolo_model = YOLO("yolov8x-seg.pt")

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

if not os.path.exists(SAM_CHECKPOINT):
    print("Downloading SAM checkpoint (~2.5GB), please wait...")
    import urllib.request
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
    print("SAM checkpoint downloaded.")

sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

YOLO_CLASSES = yolo_model.names


# ────────────────────────────────────────────────
# POLYGON EXTRACTION
# ────────────────────────────────────────────────

def mask_to_polygons(mask):
    polygons = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        poly    = approx.reshape(-1, 2).tolist()
        if len(poly) > 5:
            x, y, w, h = cv2.boundingRect(cnt)
            polygons.append({
                "points": poly,
                "area":   float(area),
                "bbox":   [int(x), int(y), int(w), int(h)]
            })
    return polygons


# ────────────────────────────────────────────────
# COCO EXPORT — POLYGON
# ────────────────────────────────────────────────

def export_coco_polygon(annotations_list, image_name, width, height):
    categories_seen = {}
    for ann in annotations_list:
        if ann["class_id"] not in categories_seen:
            categories_seen[ann["class_id"]] = ann["class_name"]

    categories = [{"id": k, "name": v} for k, v in sorted(categories_seen.items())]
    if not categories:
        categories = [{"id": 1, "name": "object"}]

    coco = {
        "images": [{"id": 1, "file_name": image_name, "width": width, "height": height}],
        "annotations": [],
        "categories": categories
    }

    for i, ann in enumerate(annotations_list, start=1):
        flat = [coord for point in ann["points"] for coord in point]
        coco["annotations"].append({
            "id":            i,
            "image_id":      1,
            "category_id":   ann["class_id"],
            "category_name": ann["class_name"],
            "segmentation":  [flat],
            "area":          ann["area"],
            "bbox":          ann["bbox"],
            "iscrowd":       0
        })

    with open("dataset/annotations/dataset.json", "w") as f:
        json.dump(coco, f, indent=4)
    return coco


# ────────────────────────────────────────────────
# COCO EXPORT — BBOX
# ────────────────────────────────────────────────

def export_coco_bbox(annotations_list, image_name, width, height):
    categories_seen = {}
    for ann in annotations_list:
        if ann["class_id"] not in categories_seen:
            categories_seen[ann["class_id"]] = ann["class_name"]

    categories = [{"id": k, "name": v} for k, v in sorted(categories_seen.items())]
    if not categories:
        categories = [{"id": 1, "name": "object"}]

    coco = {
        "images": [{"id": 1, "file_name": image_name, "width": width, "height": height}],
        "annotations": [],
        "categories": categories
    }

    for i, ann in enumerate(annotations_list, start=1):
        x1, y1, x2, y2 = ann["bbox_xyxy"]
        w = x2 - x1
        h = y2 - y1
        coco["annotations"].append({
            "id":            i,
            "image_id":      1,
            "category_id":   ann["class_id"],
            "category_name": ann["class_name"],
            "segmentation":  [],
            "area":          float(w * h),
            "bbox":          [int(x1), int(y1), int(w), int(h)],
            "iscrowd":       0
        })

    with open("dataset/annotations/dataset.json", "w") as f:
        json.dump(coco, f, indent=4)
    return coco


# ────────────────────────────────────────────────
# FEEDBACK MODELS (Pydantic)
# ────────────────────────────────────────────────

class FeedbackItem(BaseModel):
    image_name:       str
    annotation_id:    int
    class_name:       str
    mode:             str
    vote:             str                        # "up" or "down"
    ai_points:        Optional[List] = None      # original AI polygon
    corrected_points: Optional[List] = None      # user-corrected polygon
    confidence:       Optional[float] = None


class FeedbackBatch(BaseModel):
    items: List[FeedbackItem]


# ────────────────────────────────────────────────
# FEEDBACK ENDPOINT  — save thumbs up / down
# ────────────────────────────────────────────────

@app.post("/feedback")
def save_feedback(batch: FeedbackBatch):
    if not batch.items:
        raise HTTPException(status_code=400, detail="No feedback items provided.")

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        now = datetime.utcnow().isoformat()
        saved = 0

        for item in batch.items:
            if item.vote not in ("up", "down"):
                continue

            cur.execute("""
                INSERT INTO feedback
                (image_name, annotation_id, class_name, mode, vote,
                 ai_points, corrected_points, confidence, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                item.image_name,
                item.annotation_id,
                item.class_name,
                item.mode,
                item.vote,
                json.dumps(item.ai_points) if item.ai_points else None,
                json.dumps(item.corrected_points) if item.corrected_points else None,
                item.confidence or 0.0,
                now
            ))

            cur.execute("""
                INSERT INTO class_stats (class_name, total, correct, incorrect)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(class_name) DO UPDATE SET
                    total     = total + 1,
                    correct   = correct   + ?,
                    incorrect = incorrect + ?
            """, (
                item.class_name,
                1 if item.vote == "up" else 0,
                1 if item.vote == "down" else 0,
                1 if item.vote == "up" else 0,
                1 if item.vote == "down" else 0,
            ))
            saved += 1

        con.commit()
        con.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Auto-backup to HuggingFace in background
    thread = threading.Thread(target=backup_to_hf, daemon=True)
    thread.start()

    return {"status": "ok", "saved": saved}


# ────────────────────────────────────────────────
# FEEDBACK STATS ENDPOINT  — accuracy per class
# ────────────────────────────────────────────────

@app.get("/feedback/stats")
def feedback_stats():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT class_name, total, correct, incorrect FROM class_stats ORDER BY total DESC")
    rows = cur.fetchall()
    con.close()

    stats = []
    for row in rows:
        class_name, total, correct, incorrect = row
        accuracy = round((correct / total) * 100, 1) if total > 0 else 0
        stats.append({
            "class_name": class_name,
            "total":      total,
            "correct":    correct,
            "incorrect":  incorrect,
            "accuracy":   accuracy
        })

    return {"stats": stats}


# ────────────────────────────────────────────────
# DOWNLOAD FEEDBACK DB as JSON
# ────────────────────────────────────────────────

@app.get("/feedback/download")
def download_feedback():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT * FROM feedback ORDER BY id")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    con.close()

    records = []
    for row in rows:
        rec = dict(zip(cols, row))
        if rec.get("ai_points"):
            rec["ai_points"] = json.loads(rec["ai_points"])
        if rec.get("corrected_points"):
            rec["corrected_points"] = json.loads(rec["corrected_points"])
        records.append(rec)

    # Save temp JSON and serve
    out_path = "feedback/feedback_export.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=4)

    return FileResponse(out_path, media_type="application/json", filename="feedback.json")


# ────────────────────────────────────────────────
# SAVE CORRECTED ANNOTATIONS BACK TO COCO JSON
# ────────────────────────────────────────────────

class CorrectedAnnotation(BaseModel):
    image_name:  str
    image_width: int
    image_height: int
    annotations: List[dict]   # list of {id, class_id, class_name, points, type, bbox}

@app.post("/save_corrected")
def save_corrected(data: CorrectedAnnotation):
    """
    Called from frontend after user corrects polygons.
    Rebuilds dataset.json with corrected points and recalculated area/bbox.
    """
    categories_seen = {}
    coco_annotations = []

    for i, ann in enumerate(data.annotations, start=1):
        cid   = ann.get("class_id", 1)
        cname = ann.get("class_name", "object")
        categories_seen[cid] = cname

        if ann.get("type") == "polygon" and ann.get("points"):
            pts  = ann["points"]   # [[x,y], [x,y], ...]
            flat = [coord for pt in pts for coord in pt]

            # Recalculate bbox and area from corrected points
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w    = x_max - x_min
            h    = y_max - y_min
            # Shoelace formula for polygon area
            area = abs(sum(
                pts[j][0] * pts[(j+1) % len(pts)][1] - pts[(j+1) % len(pts)][0] * pts[j][1]
                for j in range(len(pts))
            )) / 2.0

            coco_annotations.append({
                "id":            i,
                "image_id":      1,
                "category_id":   cid,
                "category_name": cname,
                "segmentation":  [flat],
                "area":          round(area, 2),
                "bbox":          [round(x_min,2), round(y_min,2), round(w,2), round(h,2)],
                "iscrowd":       0
            })

        elif ann.get("type") == "bbox" and ann.get("bbox"):
            bx, by, bw, bh = ann["bbox"]
            categories_seen[cid] = cname
            coco_annotations.append({
                "id":            i,
                "image_id":      1,
                "category_id":   cid,
                "category_name": cname,
                "segmentation":  [],
                "area":          round(bw * bh, 2),
                "bbox":          [bx, by, bw, bh],
                "iscrowd":       0
            })

    categories = [{"id": k, "name": v} for k, v in sorted(categories_seen.items())]
    if not categories:
        categories = [{"id": 1, "name": "object"}]

    coco = {
        "images": [{
            "id":        1,
            "file_name": data.image_name,
            "width":     data.image_width,
            "height":    data.image_height
        }],
        "annotations": coco_annotations,
        "categories":  categories
    }

    with open("dataset/annotations/dataset.json", "w") as f:
        json.dump(coco, f, indent=4)

    return {"status": "ok", "saved": len(coco_annotations)}


# ────────────────────────────────────────────────
# DOWNLOAD COCO JSON
# ────────────────────────────────────────────────

@app.get("/download")
def download_coco():
    path = "dataset/annotations/dataset.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No annotation file found.")
    return FileResponse(path, media_type="application/json", filename="dataset.json")


# ────────────────────────────────────────────────
# ANNOTATE ENDPOINT
# ────────────────────────────────────────────────

@app.post("/annotate")
async def annotate(
    file: UploadFile,
    mode: str = Form("polygon")
):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    if mode not in ("polygon", "bbox"):
        raise HTTPException(status_code=400, detail="mode must be 'polygon' or 'bbox'.")

    upload_path      = f"uploads/{file.filename}"
    dataset_img_path = f"dataset/images/{file.filename}"

    with open(upload_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    shutil.copy(upload_path, dataset_img_path)

    image = cv2.imread(upload_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    image_rgb        = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    results     = yolo_model(upload_path)
    boxes       = results[0].boxes.xyxy.cpu().numpy()
    class_ids   = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()

    if len(boxes) == 0:
        return {
            "mode": mode, "polygons": [],
            "stats": {"total_objects": 0, "classes": {}},
            "message": "No objects detected."
        }

    annotations_list = []
    frontend_items   = []
    class_counts     = {}
    ann_id           = 0

    if mode == "polygon":
        predictor.set_image(image_rgb)
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            class_name = YOLO_CLASSES.get(int(class_id), f"class_{class_id}")
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            mask  = masks[0].astype("uint8")
            polys = mask_to_polygons(mask)
            for pd in polys:
                ann_id += 1
                annotations_list.append({
                    "points":     pd["points"],
                    "area":       pd["area"],
                    "bbox":       pd["bbox"],
                    "class_id":   int(class_id) + 1,
                    "class_name": class_name
                })
                frontend_items.append({
                    "id":         ann_id,
                    "type":       "polygon",
                    "points":     pd["points"],
                    "bbox":       pd["bbox"],
                    "class_id":   int(class_id),
                    "class_name": class_name,
                    "confidence": float(round(confidence, 3)),
                    "area":       pd["area"]
                })
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        export_coco_polygon(annotations_list, file.filename, width, height)

    else:
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            class_name      = YOLO_CLASSES.get(int(class_id), f"class_{class_id}")
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            ann_id += 1
            annotations_list.append({
                "bbox_xyxy":  [x1, y1, x2, y2],
                "class_id":   int(class_id) + 1,
                "class_name": class_name
            })
            frontend_items.append({
                "id":         ann_id,
                "type":       "bbox",
                "bbox":       [int(x1), int(y1), int(w), int(h)],
                "class_id":   int(class_id),
                "class_name": class_name,
                "confidence": float(round(confidence, 3))
            })
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        export_coco_bbox(annotations_list, file.filename, width, height)

    return {
        "mode":     mode,
        "polygons": frontend_items,
        "stats": {
            "total_objects": len(frontend_items),
            "classes":       class_counts,
            "image_size":    {"width": width, "height": height}
        }
    }
