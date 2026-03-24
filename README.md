---
title: AI Polygon Annotation Tool
emoji: 🔷
colorFrom: blue
colorTo: purple
sdk: docker
app_file: main.py
pinned: false
---

<div align="center">

<h1>PolyAnnot v2.0</h1>
<p><strong>AI-Powered Automatic Image Annotation — No Manual Labeling Needed</strong></p>

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-8A2BE2?style=flat-square)](https://ultralytics.com)
[![SAM](https://img.shields.io/badge/SAM-Meta_AI-0057FF?style=flat-square)](https://segment-anything.com)
[![HuggingFace](https://img.shields.io/badge/Live_Demo-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/0k1nx0/ai-polygon-annotation-tool)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

> 🤗 **[Try the Live Demo on Hugging Face Spaces](https://0k1nx0-ai-polygon-annotation-tool.hf.space)**

</div>

---

## What is PolyAnnot?

PolyAnnot is a web-based annotation tool that **automatically generates polygon and bounding box annotations** for any image using a state-of-the-art AI pipeline. Upload an image, choose your annotation mode, and get a ready-to-use COCO JSON dataset in seconds — no manual drawing required.

Built with **YOLOv8x** for object detection and **Meta's SAM ViT-H** for pixel-precise segmentation, it produces annotations compatible with every major training platform.

---

## What's New in v2.0

| Feature | Description |
|---|---|
| 👍 Annotation Review | Thumbs up / down feedback per detected object |
| ✏️ Polygon Correction | Drag points, add or delete vertices |
| ⬜ Bounding Box Correction | Drag corners and edges to adjust |
| ↩️ Undo / Redo | Full history with Ctrl+Z / Ctrl+Y |
| 🗄️ Training Dataset | All feedback saved to SQLite automatically |
| ☁️ Auto-Backup | Feedback synced to HuggingFace Dataset repo |
| 🔲 Sidebar Toggle | Hide/show annotation panel |
| 🔍 Zoom & Pan | Navigate large images with ease |
| 🎨 Multi-class Colors | Distinct color per class with confidence labels |
| 📦 COCO JSON Export | Corrected annotations exported in standard format |

---

## How It Works

```
Upload Image
     │
     ▼
YOLOv8x Detection ──────────► Bounding Boxes + Class Labels + Confidence
     │
     ▼  (Polygon mode only)
SAM ViT-H Segmentation ──────► Pixel-level Masks
     │
     ▼
OpenCV Contours ─────────────► Polygon Coordinates
     │
     ▼
User Review + Correction ────► Corrected Annotations saved to SQLite
     │
     ▼
COCO JSON Export ────────────► Ready-to-use Training Dataset
```

---

## Annotation Modes

| Mode | Pipeline | Speed | Output |
|---|---|---|---|
| 🔷 **Polygon** | YOLOv8 + SAM ViT-H | 15 – 60 sec | Pixel-precise polygon masks |
| ⬜ **Bounding Box** | YOLOv8 only | 3 – 15 sec | Object-level rectangles |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Object Detection | YOLOv8x (Ultralytics) |
| Instance Segmentation | SAM ViT-H (Meta AI) |
| Image Processing | OpenCV + NumPy |
| Database | SQLite |
| Frontend | HTML5 + Canvas API |
| Deployment | Docker on HuggingFace Spaces |
| Output Format | COCO JSON |

---

## Project Structure

```
ai-polygon-annotation-tool/
│
├── static/
│   └── index.html              # Frontend — Canvas UI, annotation controls
├── main.py                     # FastAPI backend — detection, segmentation, feedback
├── Dockerfile                  # Container config for HuggingFace deployment
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

> The following are created automatically at runtime:
> - `uploads/` — temporary uploaded images
> - `dataset/images/` — saved copies of annotated images
> - `dataset/annotations/dataset.json` — COCO JSON output
> - `feedback/feedback.db` — SQLite training feedback database
> - `yolov8x-seg.pt` — auto-downloaded on first run (~137MB)
> - `sam_vit_h_4b8939.pth` — auto-downloaded on first run (~2.5GB)

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/0k1nx0/ai-polygon-annotation-tool.git
cd ai-polygon-annotation-tool
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Run the server
```bash
uvicorn main:app --reload
```

### 5. Open in browser
```
http://127.0.0.1:8000
```

> **Note:** On first run, YOLOv8 (~137MB) and SAM (~2.5GB) weights are downloaded automatically.  
> GPU (CUDA) is strongly recommended for SAM segmentation speed.

---

## Output Format — COCO JSON

```json
{
  "images": [
    { "id": 1, "file_name": "image.jpg", "width": 1280, "height": 720 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "category_name": "person",
      "segmentation": [[x1, y1, x2, y2, "..."]],
      "area": 12453.0,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ],
  "categories": [
    { "id": 1, "name": "person" }
  ]
}
```

---

## Compatible Platforms

The exported COCO JSON is standard format — no conversion needed for most platforms.

| Platform | Import Method |
|---|---|
| **Roboflow** | Upload → COCO JSON |
| **CVAT** | Projects → Create Task → COCO JSON |
| **Detectron2** | Native COCO JSON support |
| **YOLOv8 Training** | Convert via `ultralytics` |
| **MMDetection** | Native COCO JSON support |
| **Hugging Face Datasets** | Load with `datasets` library |

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9+ | 3.10 |
| RAM | 8 GB | 16 GB+ |
| Disk | 4 GB free | 10 GB+ |
| GPU | Optional | CUDA (NVIDIA) |

---

## Roadmap

- [ ] YOLO `.txt` format export
- [ ] Batch image processing
- [ ] Multi-image dataset accumulation
- [ ] Active learning loop from feedback
- [ ] Model fine-tuning on collected data
- [ ] REST API for programmatic access

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Meta Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [FastAPI](https://fastapi.tiangolo.com)

---

## Developers

| Developer | Role | GitHub |
|---|---|---|
| Mohammed Abdullah | Backend, AI Pipeline, Deployment | [@0k1nx0](https://github.com/0k1nx0) |
| Karan Goyal | Frontend, UI/UX | [@karangoyal09](https://github.com/karangoyal09) |

---

<div align="center">
  <sub>Built with YOLOv8 + SAM · FastAPI · Docker · HuggingFace Spaces</sub>
</div>
