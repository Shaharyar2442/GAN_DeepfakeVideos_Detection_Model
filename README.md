<div align="center">

<img src="frontend/visionsnare/public/logo.svg" alt="VisionSnare Logo" width="80" height="80" />

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
Verdict + Confidence Score    → "Fake" or "Real"
```

---

## Project Structure

```
VisionSnare/
├── frontend/
│   └── visionsnare/          # React + Vite web application
│       ├── src/
│       │   ├── pages/        # Detect, About, HowItWorks, Pricing
│       │   └── components/   # Navbar, shared UI components
│       ├── package.json
│       └── vite.config.js
│
└── GAN_DeepfakeVideos_Detection_Model/
    ├── api/
    │   └── app.py            # FastAPI backend (POST /api/predict)
    ├── models/
    │   ├── visionsnare.py    # Main CNN-LSTM model
    │   ├── backbone.py       # CNN feature extractor
    │   ├── npr.py            # Spatial NPR module
    │   └── attention.py      # Temporal attention layer
    ├── training/
    │   └── train.py          # Training script
    ├── utils/
    │   └── predict_video.py  # Inference pipeline
    ├── checkpoints/
    │   └── model_best.pth    # Trained weights (44 MB)
    ├── model_config.json     # Model hyperparameters
    ├── requirements.txt      # Core Python dependencies
    └── requirements-api.txt  # API-specific dependencies
```

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **CUDA-capable GPU** (strongly recommended — CPU inference is very slow)
- **Git**

---

## Backend Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd VisionSnare
```

### 2. Create a Python virtual environment

```bash
cd GAN_DeepfakeVideos_Detection_Model

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) and select the version matching your system's CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 5. Start the backend API server

```bash
# From inside GAN_DeepfakeVideos_Detection_Model/
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

You should see:

```
[VisionSnare API] Checkpoint  : ...\checkpoints\model_best.pth
[VisionSnare API] Model loaded successfully ✓
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> The model loads **once** at startup. Each prediction request reuses it.

**API Endpoint:**

```
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

### 1. Open a new terminal and navigate to the frontend

```bash
cd frontend/visionsnare
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start the development server

```bash
npm run dev
```

The app will be available at **[http://localhost:5173](http://localhost:5173)**

> The Vite dev server is pre-configured to proxy `/api` requests to `http://localhost:8000`, so no CORS issues.

### 4. Build for production

```bash
npm run build
```

The output will be in `frontend/visionsnare/dist/`.

---

## Running Both Together

You need **two terminals** running simultaneously:

| Terminal | Command | Port |
|----------|---------|------|
| Backend  | `python -m uvicorn api.app:app --port 8000` (from `GAN_DeepfakeVideos_Detection_Model/`) | `8000` |
| Frontend | `npm run dev` (from `frontend/visionsnare/`) | `5173` |

Then open [http://localhost:5173](http://localhost:5173), go to **Detect**, upload a video, and click **Run VisionSnare Detection**.

---

## Training the Model

> Only needed if you want to retrain from scratch.

### 1. Prepare the dataset

Download the [FakeAVCeleb dataset](https://github.com/DASH-Lab/FakeAVCeleb) and place it inside the project. Then preprocess:

```bash
# Preprocess FaceForensics videos
python preprocess_FaceForensics_videos.py

# Or use the Jupyter notebook for FakeAVCeleb
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

Training logs and TensorBoard events are saved to `logs/`. The best checkpoint is automatically saved to `checkpoints/model_best.pth`.

### 4. Monitor with TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Swapping the Model Checkpoint

If you retrain or have a new checkpoint, no code changes needed — just update `model_config.json`:

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

Then restart the backend. See `MODEL_SWAP_GUIDE.md` for full details.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite 5 |
| Backend API | FastAPI, Uvicorn |
| ML Framework | PyTorch 2.0+ |
| Face Detection | MTCNN |
| Video Processing | OpenCV |
| Training Monitoring | TensorBoard |
| Dataset | FakeAVCeleb |

---

## Common Issues

**`No face detected` error**
The video may be too short, low-resolution, or the face is too small/occluded. Try a clearer video with visible frontal faces.

**`Video too short` error**
VisionSnare needs at least 12 frames with detectable motion. Videos under ~1 second may fail.

**Backend not reachable from frontend**
Make sure the backend is running on port `8000` before starting the frontend. Check `vite.config.js` proxy settings if using a different port.

**CUDA out of memory**
Lower-end GPUs may struggle. You can force CPU inference by setting `CUDA_VISIBLE_DEVICES=""` before starting the backend, though this will be significantly slower.

**`pip install` fails for `facenet-pytorch` / MTCNN**
Ensure your PyTorch version is installed first before `requirements.txt`, since `facenet-pytorch` depends on it.

---

## Acknowledgements

- [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) — dataset for training and evaluation
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN implementation
- [PyTorch](https://pytorch.org) — deep learning framework

---

<div align="center">

Built as a Final Year Project · VisionSnare

</div>
