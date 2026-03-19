# 🧠 AI Polygon Annotation Tool

An AI-powered automatic image annotation system that generates **polygon** and **bounding box** annotations using a hybrid deep learning pipeline — no manual labeling needed.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)](https://ultralytics.com)
[![SAM](https://img.shields.io/badge/SAM-Meta_AI-red?style=flat-square)](https://segment-anything.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 🚀 Demo

> 🤗 Try it live on **Hugging Face Spaces** → [Click Here](https://0k1nx0-ai-polygon-annotation-tool.hf.space)

---

## ✨ Features

- 📤 Upload any image via drag & drop
- 🎯 Automatic object detection using **YOLOv8x**
- 🧠 Pixel-accurate segmentation using **Meta's SAM (ViT-H)**
- 🔷 **Polygon annotation** — precise masks converted to polygon points
- ⬜ **Bounding Box annotation** — fast YOLO-only detection
- 📦 Export annotations in **COCO JSON format**
- 🎨 Color-coded overlays per class with confidence scores
- ⏱️ Live processing overlay with step-by-step progress
- 💾 Download dataset with one click

---

## 🧠 How It Works

```
Upload Image
     ↓
YOLOv8 Detection  →  Bounding Boxes + Class Labels
     ↓
SAM Segmentation  →  Pixel-level Masks  (Polygon mode only)
     ↓
OpenCV Contours   →  Polygon Coordinates
     ↓
COCO JSON Export  →  Ready-to-use Dataset
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Detection Model | YOLOv8x (Ultralytics) |
| Segmentation Model | SAM ViT-H (Meta AI) |
| Image Processing | OpenCV |
| Output Format | COCO JSON |
| Frontend | HTML + Canvas API |

---

## 🏗️ Project Structure

```
ai_polygon_tool/
│
├── main.py                    # FastAPI backend
├── static/
│   └── index.html             # Frontend UI
├── uploads/                   # Temporary uploaded images
├── dataset/
│   ├── images/                # Saved images
│   └── annotations/
│       └── dataset.json       # COCO annotations output
│
├── sam_vit_h_4b8939.pth       # SAM model weights (download separately)
├── yolov8x-seg.pt             # YOLO weights (auto-downloaded)
├── requirements.txt
└── README.md
```

---

## 🔧 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-polygon-annotation-tool.git
cd ai-polygon-annotation-tool
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Download SAM model weights
Download `sam_vit_h_4b8939.pth` from Meta's official release:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Place it in the root project folder.

### 5. Run the server
```bash
uvicorn main:app --reload
```

### 6. Open in browser
```
http://127.0.0.1:8000
```

---

## 📊 Output Format (COCO JSON)

```json
{
  "images": [{ "id": 1, "file_name": "image.jpg", "width": 640, "height": 480 }],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "category_name": "person",
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 12453.0,
      "bbox": [x, y, w, h],
      "iscrowd": 0
    }
  ],
  "categories": [{ "id": 1, "name": "person" }]
}
```

---

## 🎯 Annotation Modes

| Mode | Models Used | Speed | Accuracy |
|---|---|---|---|
| 🔷 Polygon | YOLOv8 + SAM | Slower (15–60s) | Pixel-precise |
| ⬜ Bounding Box | YOLOv8 only | Fast (5–15s) | Object-level |

---

## ⚠️ Requirements

- Python 3.9+
- 8GB+ RAM recommended
- GPU strongly recommended for SAM (CUDA)
- ~2.5GB disk space for model weights

---

## 🔮 Future Scope

These are potential improvements that could be made to extend the project:

- YOLO format export (.txt)
- Batch image processing
- Manual polygon editing
- Multi-image dataset accumulation
- Model optimization for speed

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Meta Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [FastAPI](https://fastapi.tiangolo.com)

---

## 👨‍💻 Developers

| Developer | GitHub |
|---|---|
| Mohammed Abdullah | [@0k1nx0](https://github.com/0k1nx0) |
| Karan Goyal | [@karangoyal09](https://github.com/karangoyal09) |

