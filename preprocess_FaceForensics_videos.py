import cv2
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from pathlib import Path
import warnings
from tqdm import tqdm
import glob

# --- Configuration ---

# Input directories (based on user screenshots)
# Format: (Input Path, Output Subdirectory Name)
INPUT_CONFIG = [
    (Path(r"E:\Face_Forensics\0_real\original_sequences\youtube\raw\videos"), "0_real"),
    (Path(r"E:\Face_Forensics\1_fake\manipulated_sequences\NeuralTextures\raw\videos"), "1_fake")
]

# Base output directory
OUTPUT_BASE_DIR = Path(r"E:\Processed_Face_Forensics")

# Parameters
MOTION_THRESHOLD = 0.7
OUTPUT_FACE_SIZE = (256, 256)
MIN_FRAMES_PER_VIDEO = 5
MAX_FRAMES_PER_VIDEO = 12

# Video extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# --- Initializations ---
detector = MTCNN()
warnings.filterwarnings('ignore', category=FutureWarning)

def sample_frames_with_optical_flow(video_path, threshold):
    """
    Samples frames from a video based on optical flow motion detection.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []
    
    selected_frames = []
    
    ret, prev_frame = cap.read()
    if not ret:
        # print(f"Error: Could not read the first frame of {video_path}")
        cap.release()
        return []
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    selected_frames.append(prev_frame) # Always keep the first frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate average motion score
        mean_magnitude = np.mean(magnitude)
        
        # If motion is above threshold, keep the frame
        if mean_magnitude > threshold:
            selected_frames.append(frame)
        
        prev_gray = gray

    cap.release()
    return selected_frames

def align_and_crop_faces(frames, size):
    """
    Detects, aligns, and crops faces from a list of frames using MTCNN.
    """
    processed_faces = []
    
    for frame in frames:
        # MTCNN expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = detector.detect_faces(frame_rgb)
        
        # Process only if exactly one confident face is found
        if len(results) == 1:
            keypoints = results[0]['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # Alignment
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
            
            # Re-detect face for accurate cropping on rotated image
            rotated_frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
            new_results = detector.detect_faces(rotated_frame_rgb)
            
            if len(new_results) == 1:
                x, y, width, height = new_results[0]['box']
                
                # Boundary checks
                y1, y2 = max(0, y), min(rotated_frame.shape[0], y + height)
                x1, x2 = max(0, x), min(rotated_frame.shape[1], x + width)
                
                face_crop = rotated_frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    resized_face = cv2.resize(face_crop, size, interpolation=cv2.INTER_AREA)
                    processed_faces.append(resized_face)
                    
    return processed_faces

def save_processed_frames(faces, output_folder, unique_video_name):
    """
    Saves processed face images to a target directory.
    """
    video_output_folder = output_folder / unique_video_name
    video_output_folder.mkdir(parents=True, exist_ok=True)
    
    for i, face in enumerate(faces):
        filename = f"{i:04d}.png"
        save_path = video_output_folder / filename
        cv2.imwrite(str(save_path), face)

def main():
    print("Starting Preprocessing Script...")
    print(f"Output Base Directory: {OUTPUT_BASE_DIR}")
    
    total_processed = 0
    total_skipped_frames = 0
    total_skipped_existing = 0
    
    for input_path, output_subdir in INPUT_CONFIG:
        print(f"\n--- Processing: {input_path} -> {output_subdir} ---")
        
        if not input_path.exists():
            print(f"WARNING: Input directory does not exist: {input_path}")
            continue
            
        target_output_dir = OUTPUT_BASE_DIR / output_subdir
        target_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather all video files
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            # glob is case sensitive on some OS, but windows is usually fine. 
            # Using rglob for recursive if needed, but user screenshot showed flat structure.
            # We'll use glob for flat structure as requested/observed.
            video_files.extend(list(input_path.glob(f"*{ext}")))
            
        print(f"Found {len(video_files)} videos.")
        
        # Process videos with progress bar
        for video_file in tqdm(video_files, desc=f"Processing {output_subdir}", unit="video"):
            unique_video_name = video_file.stem
            video_output_folder = target_output_dir / unique_video_name
            
            # --- Resume Feature ---
            # Check if already processed
            if video_output_folder.exists():
                existing_frames = list(video_output_folder.glob("*.png"))
                if len(existing_frames) >= MIN_FRAMES_PER_VIDEO:
                    # Already done
                    total_skipped_existing += 1
                    continue
            
            try:
                # 1. Sample Frames
                frames = sample_frames_with_optical_flow(video_file, MOTION_THRESHOLD)
                
                if len(frames) < MIN_FRAMES_PER_VIDEO:
                    total_skipped_frames += 1
                    continue
                
                # 2. Align and Crop
                faces = align_and_crop_faces(frames, OUTPUT_FACE_SIZE)
                
                if len(faces) < MIN_FRAMES_PER_VIDEO:
                    total_skipped_frames += 1
                    continue
                
                # 3. Cap Max Frames
                if len(faces) > MAX_FRAMES_PER_VIDEO:
                    faces = faces[:MAX_FRAMES_PER_VIDEO]
                
                # 4. Save
                save_processed_frames(faces, target_output_dir, unique_video_name)
                total_processed += 1
                
            except Exception as e:
                print(f"\nError processing {video_file.name}: {e}")
                continue

    print("\n" + "="*30)
    print("PROCESSING COMPLETE")
    print(f"Videos Processed: {total_processed}")
    print(f"Videos Skipped (Already Done): {total_skipped_existing}")
    print(f"Videos Skipped (Insufficient Frames/Faces): {total_skipped_frames}")
    print("="*30)

if __name__ == "__main__":
    main()
