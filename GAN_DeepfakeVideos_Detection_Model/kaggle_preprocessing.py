"""
VisionSnare — Kaggle/Colab Preprocessing Script
=================================================
This script preprocesses raw deepfake video datasets for testing with VisionSnare.
It uses the EXACT same pipeline as the local preprocessing scripts:
  1. Optical Flow frame sampling (motion threshold 0.7)
  2. Face detection + eye-landmark alignment (facenet-pytorch MTCNN)
  3. Re-detect on rotated frame → crop → resize to 256x256
  4. Save 5-12 frames per video as PNGs in per-video subfolders

Features:
  - AUTO-CHECKPOINT: Saves a zip backup to CHECKPOINT_DIR every N videos
  - RESUME: Skips already-processed videos (works after restoring from checkpoint)
  - FAST OPTICAL FLOW: Downsamples frames before computing flow (2-3x speedup)
  - GPU BATCH DETECTION: Processes multiple frames through MTCNN at once

Output structure (compatible with FakeAVCelebSequenceDataset):
  OUTPUT_BASE_DIR/
  ├── 0_real/
  │   ├── video_name_1/
  │   │   ├── 0000.png
  │   │   └── ...
  │   └── video_name_2/
  └── 1_fake/
      ├── video_name_3/
      └── ...

Usage:
  1. Edit the INPUT_CONFIG and OUTPUT_BASE_DIR below for your dataset
  2. Run: python kaggle_preprocessing.py
"""

import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
from pathlib import Path
import warnings
from tqdm import tqdm
import torch
from PIL import Image
import subprocess
import time

# ============================================================
# >>>  EDIT THIS SECTION FOR YOUR DATASET  <<<
# ============================================================

# --- Celeb-DF v2 (Default — Kaggle paths) ---
INPUT_CONFIG = [
    ("/kaggle/input/datasets/shaharyarrizwan/celeb-df-v1/Celeb-real",      "0_real"),
    ("/kaggle/input/datasets/shaharyarrizwan/celeb-df-v1/Celeb-synthesis",  "1_fake"),
]
OUTPUT_BASE_DIR = "/kaggle/working/Processed_CelebDF"

# --- Celeb-DF (Colab via Google Drive) ---
# INPUT_CONFIG = [
#     ("/content/drive/MyDrive/celeb-df-v1/Celeb-real",      "0_real"),
#     ("/content/drive/MyDrive/celeb-df-v1/Celeb-synthesis",  "1_fake"),
# ]
# OUTPUT_BASE_DIR = "/content/drive/MyDrive/Processed_CelebDF"

# --- FaceForensics++ (uncomment to use instead) ---
# INPUT_CONFIG = [
#     ("/kaggle/input/faceforensics/original_sequences/youtube/raw/videos", "0_real"),
#     ("/kaggle/input/faceforensics/manipulated_sequences/Deepfakes/raw/videos", "1_fake"),
# ]
# OUTPUT_BASE_DIR = "/kaggle/working/Processed_FaceForensics"

# --- DFDC (uncomment to use instead) ---
# INPUT_CONFIG = [
#     ("/kaggle/input/dfdc/real", "0_real"),
#     ("/kaggle/input/dfdc/fake", "1_fake"),
# ]
# OUTPUT_BASE_DIR = "/kaggle/working/Processed_DFDC"

# --- DeepfakeTIMIT (uncomment to use instead) ---
# INPUT_CONFIG = [
#     ("/kaggle/input/deepfaketimit/real", "0_real"),
#     ("/kaggle/input/deepfaketimit/fake", "1_fake"),
# ]
# OUTPUT_BASE_DIR = "/kaggle/working/Processed_DeepfakeTIMIT"

# --- Custom Dataset (uncomment and edit) ---
# INPUT_CONFIG = [
#     ("/path/to/your/real/videos/folder", "0_real"),
#     ("/path/to/your/fake/videos/folder", "1_fake"),
# ]
# OUTPUT_BASE_DIR = "/kaggle/working/Processed_Custom"

# ============================================================
# >>>  PREPROCESSING PARAMETERS  <<<
# ============================================================
MOTION_THRESHOLD = 0.7
OUTPUT_FACE_SIZE = (256, 256)
MIN_FRAMES_PER_VIDEO = 5
MAX_FRAMES_PER_VIDEO = 12
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# Optical flow speedup: downsample frames to this width before computing flow
# Set to None to disable downsampling (original resolution, slower)
OPTICAL_FLOW_RESIZE_WIDTH = None  # Disabled — use full resolution (matches original training preprocessing)

# Auto-checkpoint: save a zip backup every N videos
# Set to 0 to disable auto-checkpointing
CHECKPOINT_EVERY_N_VIDEOS = 100
CHECKPOINT_DIR = str(Path(OUTPUT_BASE_DIR).parent)  # Same parent as output
# ============================================================


# --- Initializations ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# facenet-pytorch MTCNN: keep_all=True to detect all faces, select best one ourselves
detector = MTCNN(keep_all=True, device=DEVICE)
warnings.filterwarnings('ignore')

# Pre-allocate GPU memory for faster processing
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True


def restore_from_checkpoint(output_base_dir, checkpoint_dir):
    """
    Checks if a checkpoint zip exists from a previous session.
    If found, unzips it to restore processed videos so the resume
    feature can skip them. This makes the pipeline fully resumable
    across session restarts.
    """
    zip_path = os.path.join(checkpoint_dir, "Processed_CHECKPOINT.zip")

    if not os.path.exists(zip_path):
        print("No previous checkpoint found. Starting fresh.")
        return

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n>>> FOUND CHECKPOINT: {zip_path} ({zip_size_mb:.1f} MB)")
    print(f">>> Restoring previous progress...")
    start_time = time.time()
    try:
        subprocess.run(
            ["unzip", "-o", "-q", zip_path, "-d", "/"],
            check=True, timeout=600
        )
        elapsed = time.time() - start_time
        print(f">>> RESTORED in {elapsed:.1f}s — resume will skip already-processed videos.")
    except Exception as e:
        print(f">>> WARNING: Restore failed: {e}. Starting fresh.")


def auto_checkpoint(output_base_dir, checkpoint_dir):
    """
    Creates a zip backup of the processed output directory.
    This ensures progress is saved even if the session dies.
    """
    zip_path = os.path.join(checkpoint_dir, "Processed_CHECKPOINT.zip")
    print(f"\n>>> AUTO-CHECKPOINT: Saving progress to {zip_path} ...")
    start_time = time.time()
    try:
        subprocess.run(
            ["zip", "-r", "-q", zip_path, output_base_dir],
            check=True, timeout=300
        )
        elapsed = time.time() - start_time
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f">>> CHECKPOINT SAVED: {zip_size_mb:.1f} MB ({elapsed:.1f}s)")
    except Exception as e:
        print(f">>> WARNING: Checkpoint failed: {e}")


def sample_frames_with_optical_flow(video_path, threshold):
    """
    Samples frames from a video based on optical flow motion detection.
    Only keeps frames where the average motion magnitude exceeds the threshold.

    SPEEDUP: Downsamples frames before computing optical flow (controlled by
    OPTICAL_FLOW_RESIZE_WIDTH). The original full-resolution frames are kept
    for face detection — only the flow computation uses downsampled frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []

    selected_frames = []

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []

    # Compute downsample ratio once
    if OPTICAL_FLOW_RESIZE_WIDTH and prev_frame.shape[1] > OPTICAL_FLOW_RESIZE_WIDTH:
        scale = OPTICAL_FLOW_RESIZE_WIDTH / prev_frame.shape[1]
        def downsample(img):
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        def downsample(img):
            return img

    prev_gray = cv2.cvtColor(downsample(prev_frame), cv2.COLOR_BGR2GRAY)
    selected_frames.append(prev_frame)  # Keep full-resolution frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(downsample(frame), cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow on downsampled frames (much faster)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate magnitude of flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Average motion score for this frame
        mean_magnitude = np.mean(magnitude)

        # Keep full-resolution frames with significant motion
        if mean_magnitude > threshold:
            selected_frames.append(frame)

        prev_gray = gray

    cap.release()
    return selected_frames


def batch_detect_faces(pil_images):
    """
    GPU-optimized batch face detection.
    Processes multiple images through MTCNN in a single batch on GPU
    instead of one-by-one, significantly improving GPU utilization.
    """
    all_boxes = []
    all_probs = []
    all_points = []

    BATCH_SIZE = 16
    for i in range(0, len(pil_images), BATCH_SIZE):
        batch = pil_images[i:i + BATCH_SIZE]
        batch_boxes, batch_probs, batch_points = detector.detect(batch, landmarks=True)
        for b, p, pt in zip(batch_boxes, batch_probs, batch_points):
            all_boxes.append(b)
            all_probs.append(p)
            all_points.append(pt)

    return all_boxes, all_probs, all_points


def align_and_crop_faces(frames, size):
    """
    Detects, aligns, and crops faces from a list of frames using facenet-pytorch MTCNN.
    GPU-optimized: uses batch detection to process multiple frames at once on GPU.

    Pipeline:
    Phase 1 (GPU): Batch detect faces + landmarks on all frames
    Phase 2 (CPU): Rotate frames to align eyes horizontally
    Phase 3 (GPU): Batch re-detect on rotated frames for accurate bounding boxes
    Phase 4 (CPU): Crop and resize faces to target size
    """
    processed_faces = []

    # --- Phase 1: Batch detect faces on ALL frames at once (GPU) ---
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    all_boxes, all_probs, all_points = batch_detect_faces(pil_images)

    # Collect frames that need rotation + re-detection
    frames_to_redetect = []
    alignment_data = []

    for idx, (frame, boxes, probs, points) in enumerate(zip(frames, all_boxes, all_probs, all_points)):
        if boxes is not None and len(boxes) > 0:
            best_idx = np.argmax(probs)
            landmarks = points[best_idx]
            left_eye = landmarks[0]
            right_eye = landmarks[1]

            # Calculate rotation angle (CPU - fast)
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)

            # Rotate frame (CPU - unavoidable)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

            frames_to_redetect.append(rotated_frame)
            alignment_data.append(idx)

    if not frames_to_redetect:
        return processed_faces

    # --- Phase 2: Batch re-detect on ALL rotated frames at once (GPU) ---
    rotated_pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_to_redetect]

    re_boxes = []
    re_probs = []
    BATCH_SIZE = 16
    for i in range(0, len(rotated_pil_images), BATCH_SIZE):
        batch = rotated_pil_images[i:i + BATCH_SIZE]
        batch_boxes, batch_probs = detector.detect(batch, landmarks=False)
        for b, p in zip(batch_boxes, batch_probs):
            re_boxes.append(b)
            re_probs.append(p)

    # --- Phase 3: Crop and resize (CPU) ---
    for rotated_frame, new_boxes, new_probs in zip(frames_to_redetect, re_boxes, re_probs):
        if new_boxes is not None and len(new_boxes) > 0:
            best_idx_new = np.argmax(new_probs)
            x1, y1, x2, y2 = new_boxes[best_idx_new]

            (h, w) = rotated_frame.shape[:2]
            x1, x2 = max(0, int(x1)), min(w, int(x2))
            y1, y2 = max(0, int(y1)), min(h, int(y2))

            face_crop = rotated_frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                resized_face = cv2.resize(face_crop, size, interpolation=cv2.INTER_AREA)
                processed_faces.append(resized_face)

    return processed_faces


def save_processed_frames(faces, output_folder, unique_video_name):
    """
    Saves processed face images to a target directory.
    Creates a subfolder per video: output_folder/unique_video_name/0000.png, 0001.png, ...
    """
    video_output_folder = Path(output_folder) / unique_video_name
    video_output_folder.mkdir(parents=True, exist_ok=True)

    for i, face in enumerate(faces):
        filename = f"{i:04d}.png"
        save_path = video_output_folder / filename
        cv2.imwrite(str(save_path), face)


def main():
    print("=" * 60)
    print("VisionSnare — Kaggle/Colab Preprocessing Script")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output Directory: {OUTPUT_BASE_DIR}")
    print(f"Motion Threshold: {MOTION_THRESHOLD}")
    print(f"Frame Range: {MIN_FRAMES_PER_VIDEO} - {MAX_FRAMES_PER_VIDEO}")
    print(f"Optical Flow Downscale: {OPTICAL_FLOW_RESIZE_WIDTH or 'Disabled (full res)'}")
    print(f"Auto-Checkpoint Every: {CHECKPOINT_EVERY_N_VIDEOS} videos")
    print("=" * 60)

    # --- Auto-Restore from previous checkpoint ---
    restore_from_checkpoint(OUTPUT_BASE_DIR, CHECKPOINT_DIR)

    total_processed = 0
    total_skipped_frames = 0
    total_skipped_existing = 0
    videos_since_checkpoint = 0

    for input_path_str, output_subdir in INPUT_CONFIG:
        input_path = Path(input_path_str)
        print(f"\n--- Processing: {input_path} -> {output_subdir} ---")

        if not input_path.exists():
            print(f"WARNING: Input directory does not exist: {input_path}")
            print(f"  Make sure the dataset is uploaded and the path is correct.")
            continue

        target_output_dir = Path(OUTPUT_BASE_DIR) / output_subdir
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # Gather all video files (flat search in the input directory)
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(list(input_path.glob(f"*{ext}")))

        print(f"Found {len(video_files)} videos.")

        if len(video_files) == 0:
            # Try recursive search in case videos are in subdirectories
            print("No videos found in flat search, trying recursive search...")
            for ext in VIDEO_EXTENSIONS:
                video_files.extend(list(input_path.rglob(f"*{ext}")))
            print(f"Found {len(video_files)} videos (recursive).")

        # Process videos with progress bar
        for video_file in tqdm(video_files, desc=f"Processing {output_subdir}", unit="video"):
            unique_video_name = video_file.stem
            video_output_folder = target_output_dir / unique_video_name

            # --- Resume Feature ---
            # Skip already processed videos
            if video_output_folder.exists():
                existing_frames = list(video_output_folder.glob("*.png"))
                if len(existing_frames) >= MIN_FRAMES_PER_VIDEO:
                    total_skipped_existing += 1
                    continue

            try:
                # Step 1: Sample frames based on optical flow motion
                frames = sample_frames_with_optical_flow(video_file, MOTION_THRESHOLD)

                if len(frames) < MIN_FRAMES_PER_VIDEO:
                    total_skipped_frames += 1
                    continue

                # Step 2: Detect, align, and crop faces
                faces = align_and_crop_faces(frames, OUTPUT_FACE_SIZE)

                if len(faces) < MIN_FRAMES_PER_VIDEO:
                    total_skipped_frames += 1
                    continue

                # Step 3: Cap at maximum frames
                if len(faces) > MAX_FRAMES_PER_VIDEO:
                    faces = faces[:MAX_FRAMES_PER_VIDEO]

                # Step 4: Save processed frames
                save_processed_frames(faces, target_output_dir, unique_video_name)
                total_processed += 1
                videos_since_checkpoint += 1

                # --- Auto-Checkpoint ---
                if CHECKPOINT_EVERY_N_VIDEOS > 0 and videos_since_checkpoint >= CHECKPOINT_EVERY_N_VIDEOS:
                    auto_checkpoint(OUTPUT_BASE_DIR, CHECKPOINT_DIR)
                    videos_since_checkpoint = 0

            except Exception as e:
                print(f"\nError processing {video_file.name}: {e}")
                continue

    # --- Final Checkpoint ---
    if total_processed > 0 and CHECKPOINT_EVERY_N_VIDEOS > 0:
        print("\nCreating final checkpoint...")
        auto_checkpoint(OUTPUT_BASE_DIR, CHECKPOINT_DIR)

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Videos Processed:                    {total_processed}")
    print(f"Videos Skipped (Already Done):       {total_skipped_existing}")
    print(f"Videos Skipped (Insufficient Faces): {total_skipped_frames}")
    print("=" * 60)

    # Print output directory contents for verification
    output_path = Path(OUTPUT_BASE_DIR)
    if output_path.exists():
        print(f"\nOutput directory structure:")
        for class_dir in sorted(output_path.iterdir()):
            if class_dir.is_dir():
                num_videos = len([d for d in class_dir.iterdir() if d.is_dir()])
                print(f"  {class_dir.name}/  ->  {num_videos} video folders")


if __name__ == "__main__":
    main()
