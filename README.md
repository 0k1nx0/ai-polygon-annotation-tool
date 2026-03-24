<div align="center">

# 🔷 PolyAnnot v2.0

### AI-Powered Automatic Image Annotation — No Manual Labeling Needed

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-7c3aed?style=flat-square)](https://ultralytics.com)
[![SAM](https://img.shields.io/badge/SAM-Meta_AI-0057FF?style=flat-square)](https://segment-anything.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

> 🤗 Try it live on **Hugging Face Spaces** → **[Click Here](https://0k1nx0-ai-polygon-annotation-tool.hf.space)**

</div>

---

## 📌 What is PolyAnnot?

**PolyAnnot** is a free, open-source web app that automatically annotates your images using AI.

Upload an image → the AI detects every object → draws precise polygon outlines → exports a ready-to-use **COCO JSON dataset**. No drawing, no clicking, no manual work.

It uses two of the most powerful open-source vision models:
- 🎯 **YOLOv8x** by Ultralytics — detects objects and their locations
- 🧠 **SAM ViT-H** by Meta AI — draws pixel-perfect masks around each object

> Perfect for students, researchers, and developers building computer vision projects.

---

## ✨ What's New in v2.0

| Feature | Description |
|---|---|
| 👍 Annotation Review | Thumbs up / down per detected object |
| ✏️ Polygon Correction | Drag, add, or delete polygon points |
| ⬜ BBox Correction | Drag corners and edges to adjust |
| ↩️ Undo / Redo | Full history — Ctrl+Z and Ctrl+Y |
| 🗄️ Training Dataset | Feedback saved to SQLite automatically |
| ☁️ Auto Cloud Backup | Feedback synced to HuggingFace Dataset repo |
| 🔲 Sidebar Toggle | Clean workspace when you need it |
| 🔍 Zoom & Pan | Navigate large images comfortably |
| 🎨 Multi-class Colors | Unique color per class with confidence labels |
| 📦 COCO JSON Export | Corrected annotations in standard format |

---

## 🧠 How It Works

```
Upload Image
     │
     ▼
YOLOv8x Detection  ──►  Bounding Boxes + Class Labels + Confidence
     │
     ▼  (Polygon mode only)
SAM ViT-H Segmentation  ──►  Pixel-level Masks
     │
     ▼
OpenCV Contours  ──►  Polygon Coordinates
     │
     ▼
Review & Correct  ──►  Thumbs up/down + Edit Points
     │
     ▼
COCO JSON Export  ──►  Ready-to-use Training Dataset
```

---

## 🎯 Annotation Modes

| Mode | Models Used | Speed | Accuracy | Best For |
|---|---|---|---|---|
| 🔷 **Polygon** | YOLOv8 + SAM | 15 – 60s | Pixel-precise | Segmentation training |
| ⬜ **Bounding Box** | YOLOv8 only | 3 – 15s | Object-level | Detection training |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Object Detection | YOLOv8x (Ultralytics) |
| Segmentation | SAM ViT-H (Meta AI) |
| Image Processing | OpenCV + NumPy |
| Database | SQLite |
| Frontend | HTML5 + Canvas API |
| Deployment | Docker on HuggingFace Spaces |
| Output Format | COCO JSON |

---

## 🏗️ Project Structure

```
ai-polygon-annotation-tool/
│
├── static/
│   └── index.html           # Frontend — Canvas UI, annotation controls
├── main.py                  # Backend — FastAPI, AI models, feedback API
├── Dockerfile               # Container config for HuggingFace deployment
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore
```

> 📝 The following are created automatically at runtime and are **not** stored in the repo:
> - `uploads/` → temporary uploaded images
> - `dataset/images/` → saved annotated images
> - `dataset/annotations/dataset.json` → COCO JSON output
> - `feedback/feedback.db` → SQLite feedback database
> - `yolov8x-seg.pt` → auto-downloaded on first run (~137 MB)
> - `sam_vit_h_4b8939.pth` → auto-downloaded on first run (~2.5 GB)

---

## 🔧 Local Setup

> **Prerequisites:** Python 3.9+, Git, ~4 GB free disk space

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

### 5. Open in your browser
```
http://127.0.0.1:8000
```

> ⚠️ On first run, YOLOv8 (~137 MB) and SAM (~2.5 GB) weights are downloaded automatically. Allow a few minutes on first launch only.

---

## 📊 Output Format — COCO JSON

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
      "segmentation": [[120, 80, 135, 95, 150, 110]],
      "area": 4820.5,
      "bbox": [120, 80, 200, 300],
      "iscrowd": 0
    }
  ],
  "categories": [
    { "id": 1, "name": "person" }
  ]
}
```

---

## 🔌 Platform Compatibility

The exported COCO JSON works directly with every major ML platform — no conversion needed.

| Platform | How to Import |
|---|---|
| **Roboflow** | Upload → Select COCO JSON format |
| **CVAT** | Projects → Create Task → Upload COCO JSON |
| **Detectron2** | Native COCO JSON support out of the box |
| **MMDetection** | Native COCO JSON support out of the box |
| **YOLOv8 Training** | Convert via `ultralytics` COCO → YOLO utility |
| **Hugging Face Datasets** | Load directly with the `datasets` library |

> 💡 Standard COCO format — no conversion needed for most platforms.

---

## ⚠️ System Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10 |
| RAM | 8 GB | 16 GB+ |
| Disk | 4 GB | 10 GB+ |
| GPU | Not required | NVIDIA CUDA (10x faster) |
| OS | Windows / Mac / Linux | Any |

---

## 🔮 Roadmap

- [x] YOLOv8 + SAM polygon annotation
- [x] Bounding box mode
- [x] Annotation review system
- [x] Polygon and bbox correction tools
- [x] Undo / Redo
- [x] COCO JSON export
- [x] SQLite feedback database
- [x] HuggingFace auto-backup
- [ ] YOLO `.txt` format export
- [ ] Batch image processing
- [ ] Multi-image dataset accumulation
- [ ] Active learning from user feedback
- [ ] Fine-tuning pipeline on collected data

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute for personal and commercial projects.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Meta Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [FastAPI](https://fastapi.tiangolo.com)

---

## 👨‍💻 Developers

| Developer | Role | GitHub |
|---|---|---|
| **Mohammed Abdullah** | Backend · AI Pipeline · Deployment | [@0k1nx0](https://github.com/0k1nx0) |
| **Karan Goyal** | Frontend · UI/UX | [@karangoyal09](https://github.com/karangoyal09) |

---

<div align="center">
  <sub>⭐ If this project helped you, please give it a star on GitHub!</sub>
  <br/>
  <sub>Built with ❤️ using YOLOv8 · SAM · FastAPI · Docker</sub>
</div>
