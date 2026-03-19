from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import cv2
import json
import os
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
predictor = SamPredictor(sam)


# ------------------------
# POLYGON FUNCTION
# ------------------------

def mask_to_polygons(mask):

    polygons = []

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 1000:
            continue

        epsilon = 0.002 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)

        poly = approx.reshape(-1,2).tolist()

        if len(poly) > 5:
            polygons.append(poly)

    return polygons


# ------------------------
# COCO EXPORT
# ------------------------

def export_coco(polygons,image_name,width,height):

    coco = {
        "images":[
            {
                "id":1,
                "file_name":image_name,
                "width":width,
                "height":height
            }
        ],
        "annotations":[],
        "categories":[
            {"id":1,"name":"object"}
        ]
    }

    ann_id = 1

    for poly in polygons:

        flat = [coord for point in poly for coord in point]

        coco["annotations"].append({
            "id":ann_id,
            "image_id":1,
            "category_id":1,
            "segmentation":[flat],
            "area":0,
            "bbox":[0,0,0,0],
            "iscrowd":0
        })

        ann_id += 1

    with open("dataset/annotations/dataset.json","w") as f:
        json.dump(coco,f,indent=4)


# ------------------------
# ANNOTATE API
# ------------------------

@app.post("/annotate")
async def annotate(file: UploadFile):

    upload_path = f"uploads/{file.filename}"
    dataset_image_path = f"dataset/images/{file.filename}"

    with open(upload_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    shutil.copy(upload_path,dataset_image_path)

    image = cv2.imread(upload_path)
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    height,width,_ = image.shape

    # YOLO DETECTION
    results = yolo_model(upload_path)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    predictor.set_image(image_rgb)

    polygons = []

    for box in boxes:

        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False
        )

        mask = masks[0].astype("uint8")

        polys = mask_to_polygons(mask)

        polygons.extend(polys)

    export_coco(polygons,file.filename,width,height)

    return {"polygons":polygons}