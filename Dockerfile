FROM python:3.9-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Copy app source
COPY . .

# Create required directories
RUN mkdir -p uploads dataset/images dataset/annotations static

# Download SAM weights (ViT-H ~2.4GB) at build time
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# YOLOv8 weights auto-download on first run
# Pre-download to avoid timeout on first request
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x-seg.pt')"

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run FastAPI on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
