from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import cv2
import json
import os
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/annotations", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")


# ------------------------
# LOAD MODELS
# ------------------------

yolo_model = YOLO("yolov8x-seg.pt")

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

YOLO_CLASSES = yolo_model.names  # {0: 'person', 1: 'bicycle', ...}


# ------------------------
# POLYGON EXTRACTION
# ------------------------

def mask_to_polygons(mask):
    polygons = []

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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


# ------------------------
# COCO EXPORT — POLYGON
# ------------------------

def export_coco_polygon(annotations_list, image_name, width, height):
    categories_seen = {}
    for ann in annotations_list:
        cid   = ann["class_id"]
        cname = ann["class_name"]
        if cid not in categories_seen:
            categories_seen[cid] = cname

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


# ------------------------
# COCO EXPORT — BBOX
# ------------------------

def export_coco_bbox(annotations_list, image_name, width, height):
    categories_seen = {}
    for ann in annotations_list:
        cid   = ann["class_id"]
        cname = ann["class_name"]
        if cid not in categories_seen:
            categories_seen[cid] = cname

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
        w    = x2 - x1
        h    = y2 - y1
        area = float(w * h)

        coco["annotations"].append({
            "id":            i,
            "image_id":      1,
            "category_id":   ann["class_id"],
            "category_name": ann["class_name"],
            "segmentation":  [],
            "area":          area,
            "bbox":          [int(x1), int(y1), int(w), int(h)],
            "iscrowd":       0
        })

    with open("dataset/annotations/dataset.json", "w") as f:
        json.dump(coco, f, indent=4)

    return coco


# ------------------------
# DOWNLOAD ENDPOINT
# ------------------------

@app.get("/download")
def download_coco():
    path = "dataset/annotations/dataset.json"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No annotation file found. Please annotate an image first."
        )
    return FileResponse(path, media_type="application/json", filename="dataset.json")


# ------------------------
# ANNOTATE ENDPOINT
# mode: "polygon" or "bbox"
# ------------------------

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

    # YOLO detection (shared for both modes)
    results     = yolo_model(upload_path)
    boxes       = results[0].boxes.xyxy.cpu().numpy()
    class_ids   = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()

    if len(boxes) == 0:
        return {
            "mode":     mode,
            "polygons": [],
            "stats":    {"total_objects": 0, "classes": {}},
            "message":  "No objects detected in this image."
        }

    annotations_list = []
    frontend_items   = []
    class_counts     = {}

    # ── POLYGON mode ──────────────────────────────────
    if mode == "polygon":
        predictor.set_image(image_rgb)

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            class_name = YOLO_CLASSES.get(int(class_id), f"class_{class_id}")

            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            mask  = masks[0].astype("uint8")
            polys = mask_to_polygons(mask)

            for pd in polys:
                annotations_list.append({
                    "points":     pd["points"],
                    "area":       pd["area"],
                    "bbox":       pd["bbox"],
                    "class_id":   int(class_id) + 1,
                    "class_name": class_name
                })
                frontend_items.append({
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

    # ── BBOX mode ─────────────────────────────────────
    else:
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            class_name      = YOLO_CLASSES.get(int(class_id), f"class_{class_id}")
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            annotations_list.append({
                "bbox_xyxy":  [x1, y1, x2, y2],
                "class_id":   int(class_id) + 1,
                "class_name": class_name
            })
            frontend_items.append({
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