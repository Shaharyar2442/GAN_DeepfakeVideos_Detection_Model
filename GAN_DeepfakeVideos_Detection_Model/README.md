<div align="center">

# VisionSnare

### AI-Powered Deepfake Video Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)

*Upload a video. Get the truth.*

</div>

---

## What is VisionSnare?

VisionSnare is a deepfake video detection system built as a Final Year Project. It uses a custom **CNN-LSTM neural network** trained on the FakeAVCeleb dataset to determine whether a video is authentic or AI-generated. The full pipeline runs locally — no cloud, no subscriptions, no data leaving your machine.

Upload a video → VisionSnare extracts faces, analyzes temporal patterns, and returns a **Deepfake** or **Authentic** verdict with a confidence score.

---

## How It Works

```
Video Upload
     │
     ▼
Optical Flow Frame Sampling   ← Motion-based keyframe selection
     │
     ▼
MTCNN Face Detection          ← Face alignment & cropping (256×256)
     │
     ▼
Spatial NPR Feature Map       ← 2×2 Neighbor Pixel Relationship grids
     │
     ▼
CNN Backbone                  ← Spatial feature extraction per frame
     │
     ▼
LSTM (hidden_dim=512)         ← 11-frame temporal sequence modelling
     │
     ▼
Verdict + Confidence Score    → "Deepfake" or "Authentic"
```

---

## Project Structure

This repository contains the **model & backend API**. The React frontend lives alongside it as a sibling directory.

```
parent-directory/
│
├── GAN_DeepfakeVideos_Detection_Model/   ← this repo
│   ├── api/
│   │   └── app.py            # FastAPI backend (POST /api/predict)
│   ├── models/
│   │   ├── visionsnare.py    # Main CNN-LSTM model
│   │   ├── backbone.py       # CNN feature extractor
│   │   ├── npr.py            # Spatial NPR module
│   │   └── attention.py      # Temporal attention layer
│   ├── training/
│   │   └── train.py          # Training script
│   ├── utils/
│   │   └── predict_video.py  # Inference pipeline
│   ├── checkpoints/
│   │   └── model_best.pth    # Trained weights (~44 MB)
│   ├── model_config.json     # Model hyperparameters
│   ├── requirements.txt      # Core Python dependencies
│   └── requirements-api.txt  # API-specific dependencies
│
└── frontend/
    └── visionsnare/          # React + Vite web application
        ├── src/
        │   ├── pages/        # Detect, About, HowItWorks, Pricing
        │   └── components/   # Navbar, shared UI components
        ├── package.json
        └── vite.config.js
```

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **CUDA-capable GPU** (strongly recommended — CPU inference is very slow)

---

## Backend Setup

### 1. Create a Python virtual environment

```bash
cd GAN_DeepfakeVideos_Detection_Model

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install PyTorch

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) and select the version that matches your CUDA. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 4. Start the backend API server

```bash
# Run from inside GAN_DeepfakeVideos_Detection_Model/
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

You should see:

```
[VisionSnare API] Checkpoint  : ...\checkpoints\model_best.pth
[VisionSnare API] Model loaded successfully ✓
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> The model loads **once** at startup and is reused for every request.

**API Endpoint:**

```http
POST http://localhost:8000/api/predict
Content-Type: multipart/form-data

Body: video=<file>

Response:
{
  "verdict": "fake" | "real",
  "confidence": 94.2,
  "raw_score": 0.942
}
```

---

## Frontend Setup

### 1. Navigate to the frontend directory

```bash
cd ../frontend/visionsnare
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start the development server

```bash
npm run dev
```

Open **[http://localhost:5173](http://localhost:5173)** in your browser.

> The Vite dev server proxies `/api` requests to `http://localhost:8000` automatically — no CORS configuration needed.

### 4. Build for production

```bash
npm run build
# Output goes to frontend/visionsnare/dist/
```

---

## Running Both Together

You need **two terminals** open at the same time:

| Terminal | Directory | Command | Port |
|----------|-----------|---------|------|
| **Backend** | `GAN_DeepfakeVideos_Detection_Model/` | `python -m uvicorn api.app:app --port 8000` | `8000` |
| **Frontend** | `frontend/visionsnare/` | `npm run dev` | `5173` |

Then open [http://localhost:5173](http://localhost:5173), go to **Detect**, upload a video, and click **Run VisionSnare Detection**.

---

## Training the Model

> Only needed if you want to retrain from scratch.

### 1. Prepare the dataset

Download [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) and preprocess it:

```bash
# Preprocess videos
python preprocess_FaceForensics_videos.py

# Or use the Jupyter notebook
jupyter lab FakeAvCelebPreprocessing.ipynb
```

### 2. Set the data path

Open `training/train.py` and update line 17:

```python
DATA_ROOT = r"C:\path\to\your\Processed_FakeAVCeleb"
```

### 3. Run training

```bash
python training/train.py
```

Logs are saved to `logs/`. The best checkpoint is automatically saved to `checkpoints/model_best.pth`.

### 4. Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Swapping the Model Checkpoint

No code changes needed — just edit `model_config.json`:

```json
{
    "model_name": "VisionSnare",
    "checkpoint": "checkpoints/your_new_checkpoint.pth",
    "lstm_hidden_dim": 512,
    "sequence_length": 12,
    "image_size": [256, 256],
    "motion_threshold": 0.7
}
```

Restart the backend and you're done. See [`MODEL_SWAP_GUIDE.md`](MODEL_SWAP_GUIDE.md) for full details.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite 5 |
| Backend API | FastAPI, Uvicorn |
| ML Framework | PyTorch 2.0+ |
| Face Detection | MTCNN (facenet-pytorch) |
| Video Processing | OpenCV |
| Training Monitoring | TensorBoard |
| Dataset | FakeAVCeleb |

---

## Common Issues

**`No face detected` error**
The video may be too short, low-resolution, or the face is occluded. Use a clear video with a visible frontal face.

**`Video too short` error**
VisionSnare needs at least 12 frames with detectable motion. Videos under ~1 second may fail.

**Backend not reachable from frontend**
Make sure the backend is running on port `8000` before starting the frontend.

**CUDA out of memory**
Force CPU mode by setting `CUDA_VISIBLE_DEVICES=""` before starting the backend. This will be significantly slower.

**`pip install` fails for `facenet-pytorch` / MTCNN**
Install PyTorch before running `pip install -r requirements.txt` since `facenet-pytorch` depends on it.

---

## Acknowledgements

- [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) — training dataset
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN implementation
- [PyTorch](https://pytorch.org) — deep learning framework

---

<div align="center">

Built as a Final Year Project &nbsp;·&nbsp; VisionSnare

</div>
