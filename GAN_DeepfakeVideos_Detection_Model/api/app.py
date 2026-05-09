"""
VisionSnare API — FastAPI backend for deepfake video detection.

Wraps the existing preprocessing pipeline (optical flow, MTCNN alignment)
and VisionSnare model inference into a single POST endpoint.

Usage:
    cd d:/Fyp_Project/GAN_DeepfakeVideos_Detection_Model
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import uuid
import shutil
import tempfile
import warnings

try:
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    AI_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[VisionSnare API] WARNING: Missing AI packages. Video detection will fail. ({e})")
    AI_MODULES_AVAILABLE = False

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so model imports resolve correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if AI_MODULES_AVAILABLE:
    from facenet_pytorch import MTCNN
    from models.visionsnare import VisionSnare

# ---------------------------------------------------------------------------
# Configuration — loaded from model_config.json (edit THAT file to swap models)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(PROJECT_ROOT, "model_config.json")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"model_config.json not found at {CONFIG_PATH}. "
        "Please create it (see MODEL_SWAP_GUIDE.md)."
    )

with open(CONFIG_PATH, "r") as _f:
    _cfg = json.load(_f)

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, _cfg["checkpoint"])
SEQUENCE_LENGTH = _cfg.get("sequence_length", 12)
IMAGE_SIZE = tuple(_cfg.get("image_size", [256, 256]))
MOTION_THRESHOLD = _cfg.get("motion_threshold", 0.7)
LSTM_HIDDEN_DIM = _cfg.get("lstm_hidden_dim", 512)
if AI_MODULES_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# One-time model loading (singleton, avoids ~2 s reload per request)
# ---------------------------------------------------------------------------
print(f"[VisionSnare API] Config loaded from {CONFIG_PATH}")
print(f"[VisionSnare API] Checkpoint  : {CHECKPOINT_PATH}")
print(f"[VisionSnare API] Seq length  : {SEQUENCE_LENGTH}")
print(f"[VisionSnare API] LSTM hidden : {LSTM_HIDDEN_DIM}")
print(f"[VisionSnare API] Loading model on {DEVICE} …")

if AI_MODULES_AVAILABLE:
    _model = VisionSnare(lstm_hidden_dim=LSTM_HIDDEN_DIM).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {CHECKPOINT_PATH}. "
            "Please ensure the checkpoint file exists at the path specified in model_config.json."
        )

    _ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    _model.load_state_dict(_ckpt["model_state_dict"])
    _model.eval()
    print("[VisionSnare API] Model loaded successfully ✓")

    # MTCNN face detector (also singleton)
    _detector = MTCNN(keep_all=True, device=DEVICE)
    print("[VisionSnare API] MTCNN face detector ready ✓")
else:
    print("[VisionSnare API] Skipping model loading due to missing dependencies.")

# ---------------------------------------------------------------------------
# Pre-processing helpers (copied from utils/predict_video.py — no changes)
# ---------------------------------------------------------------------------

def _sample_frames_optical_flow(video_path: str, threshold: float, target_count: int = 12):
    """Sample frames with significant motion via dense optical flow."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames, all_frames = [], []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(prev_frame)
    all_frames.append(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if np.mean(magnitude) > threshold:
            frames.append(frame)
            if len(frames) >= target_count:
                break
        prev_gray = gray

    cap.release()

    if len(frames) >= target_count:
        return frames[:target_count]

    # Fall back to uniform sampling if not enough motion frames
    if len(all_frames) < target_count:
        return []
    indices = np.linspace(0, len(all_frames) - 1, target_count, dtype=int)
    return [all_frames[i] for i in indices]


def _align_and_crop_faces(frames, detector, size):
    """Detect, align and crop faces using facenet-pytorch MTCNN."""
    processed = []
    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs, points = detector.detect(img, landmarks=True)

        if boxes is None:
            continue

        best = np.argmax(probs)
        left_eye, right_eye = points[best][0], points[best][1]
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

        img_rot = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        new_boxes, new_probs = detector.detect(img_rot, landmarks=False)

        if new_boxes is not None:
            best_new = np.argmax(new_probs)
            x1, y1, x2, y2 = new_boxes[best_new]
            x1, x2 = max(0, int(x1)), min(w, int(x2))
            y1, y2 = max(0, int(y1)), min(h, int(y2))
            crop = rotated[y1:y2, x1:x2]
            if crop.size > 0:
                processed.append(cv2.resize(crop, size, interpolation=cv2.INTER_AREA))

    return processed


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
from api.auth import auth_router, get_current_user
from api.database import get_history_by_username, add_history_entry

app = FastAPI(title="VisionSnare API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])

from fastapi.staticfiles import StaticFiles
import os

# Ensure the dist directory exists to prevent startup crash if frontend isn't built yet
dist_path = os.path.join(PROJECT_ROOT, "frontend", "visionsnare", "dist")
if os.path.exists(dist_path):
    app.mount("/", StaticFiles(directory=dist_path, html=True), name="frontend")
else:
    print(f"[VisionSnare] WARNING: Frontend dist folder not found at {dist_path}. Please run 'npm run build'.")

class HistoryEntry(BaseModel):
    id: int
    filename: str
    date: str
    size: str
    verdict: str
    confidence: float
    duration: str

@app.get("/api/history")
async def read_history(current_user: dict = Depends(get_current_user)):
    history = get_history_by_username(current_user["username"])
    return {"history": history}

@app.post("/api/history")
async def append_history(entry: HistoryEntry, current_user: dict = Depends(get_current_user)):
    entry_dict = entry.model_dump()
    entry_dict["username"] = current_user["username"]
    add_history_entry(entry_dict)
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/api/predict")
async def predict(video: UploadFile = File(...)):
    """
    Accept a video upload, run the full VisionSnare pipeline, and return
    the deepfake detection verdict with confidence score.
    """
    if not AI_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="AI modules (torch/cv2) are missing. Detection pipeline offline.",
        )

    # ── 1. Validate file type ──────────────────────────────────────────
    allowed = (".mp4", ".mov", ".avi", ".mkv")
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    # ── 2. Save to temp file ──────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="visionsnare_")
    video_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{ext}")
    try:
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        # ── 3. Frame sampling (optical flow) ──────────────────────────
        frames = _sample_frames_optical_flow(video_path, MOTION_THRESHOLD, SEQUENCE_LENGTH)
        if not frames:
            raise HTTPException(
                status_code=422,
                detail="Could not extract enough frames. The video may be too short or corrupted.",
            )

        # ── 4. Face detection & alignment (MTCNN) ────────────────────
        faces = _align_and_crop_faces(frames, _detector, IMAGE_SIZE)
        if len(faces) < SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"Could not detect a face in enough frames. Detected {len(faces)}/{SEQUENCE_LENGTH} faces.",
            )
        faces = faces[:SEQUENCE_LENGTH]

        # ── 5. Build tensor ───────────────────────────────────────────
        transform = transforms.ToTensor()
        tensors = []
        for face in faces:
            img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            tensors.append(transform(img))
        input_tensor = torch.stack(tensors).unsqueeze(0).to(DEVICE)

        # ── 6. Model inference ────────────────────────────────────────
        with torch.no_grad():
            logits = _model(input_tensor)
            prob = torch.sigmoid(logits).item()

        verdict = "fake" if prob > 0.5 else "real"
        confidence = round((prob if prob > 0.5 else 1 - prob) * 100, 2)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "raw_score": round(prob, 4),
        }

    finally:
        # ── 7. Cleanup temp files ─────────────────────────────────────
        shutil.rmtree(tmp_dir, ignore_errors=True)
