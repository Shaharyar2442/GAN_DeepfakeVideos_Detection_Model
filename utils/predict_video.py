import cv2
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
# CHANGED: Use facenet-pytorch (Pure PyTorch) instead of TensorFlow MTCNN
from facenet_pytorch import MTCNN
import warnings

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.visionsnare import VisionSnare

# --- CONFIGURATION ---
VIDEO_INPUT_FOLDER = "video_analysis"
CHECKPOINT_PATH = os.path.join("checkpoints", "model_best.pth")
SEQUENCE_LENGTH = 12
IMAGE_SIZE = (256, 256)
MOTION_THRESHOLD = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. PREPROCESSING FUNCTIONS ---

def sample_frames_with_optical_flow(video_path, threshold, target_count=12):
    """Samples frames based on motion using Optical Flow."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []
    
    frames = []
    all_frames = [] 
    
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
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        
        if mean_magnitude > threshold:
            frames.append(frame)
        
        prev_gray = gray

    cap.release()
    
    if len(frames) >= target_count:
        selected_frames = frames[:target_count]
    else:
        print(f"Warning: Only found {len(frames)} motion frames. Padding with uniform sampling.")
        total_frames = len(all_frames)
        if total_frames < target_count:
            print(f"Error: Video too short. Has {total_frames} frames, need {target_count}.")
            return []
        indices = np.linspace(0, total_frames - 1, target_count, dtype=int)
        selected_frames = [all_frames[i] for i in indices]
        
    return selected_frames

def align_and_crop_faces(frames, detector, size):
    """
    Detects, aligns, and crops faces using facenet-pytorch.
    """
    processed_faces = []
    
    for frame in frames:
        # Convert to PIL for facenet-pytorch
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect faces and landmarks
        # boxes: [N, 4], probs: [N], points: [N, 5, 2]
        boxes, probs, points = detector.detect(img, landmarks=True)
        
        if boxes is not None:
            # Select the face with the highest probability
            best_idx = np.argmax(probs)
            
            # Get landmarks for alignment (points[i] = [left_eye, right_eye, nose, mouth_l, mouth_r])
            landmarks = points[best_idx]
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Alignment Calculation
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            (h, w) = frame.shape[:2]
            center = (w // 2, h // 2)
            
            # Rotate frame
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
            
            # Re-detect on rotated frame for accurate crop
            img_rotated = Image.fromarray(cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB))
            new_boxes, new_probs = detector.detect(img_rotated, landmarks=False)
            
            if new_boxes is not None:
                best_idx_new = np.argmax(new_probs)
                x1, y1, x2, y2 = new_boxes[best_idx_new]
                
                # Convert to int and crop
                x1, x2 = max(0, int(x1)), min(w, int(x2))
                y1, y2 = max(0, int(y1)), min(h, int(y2))
                
                face_crop = rotated_frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    resized_face = cv2.resize(face_crop, size, interpolation=cv2.INTER_AREA)
                    processed_faces.append(resized_face)
    
    return processed_faces

def preprocess_video_pipeline(video_path, output_folder):
    """Runs the full Optical Flow -> Align -> Crop pipeline."""
    print(f"Processing video: {video_path}...")
    
    # 1. Sample Frames
    frames = sample_frames_with_optical_flow(video_path, MOTION_THRESHOLD, SEQUENCE_LENGTH)
    if not frames:
        return False
        
    # 2. Detect & Align (Initialize MTCNN on GPU if available)
    detector = MTCNN(keep_all=True, device=DEVICE)
    faces = align_and_crop_faces(frames, detector, IMAGE_SIZE)
    
    if len(faces) < SEQUENCE_LENGTH:
        print(f"Error: Could not extract {SEQUENCE_LENGTH} aligned faces. Found {len(faces)}.")
        return False
    
    faces = faces[:SEQUENCE_LENGTH]
    
    # 3. Save Frames
    os.makedirs(output_folder, exist_ok=True)
    for i, face in enumerate(faces):
        filename = f"{i:04d}.png"
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, face)
        
    print(f"-> Saved {len(faces)} processed frames to {output_folder}")
    return True

# --- 2. PREDICTION FUNCTION ---

def predict(processed_folder):
    frames = []
    transform = transforms.ToTensor() 
    
    files = sorted(os.listdir(processed_folder))
    valid_files = [f for f in files if f.endswith('.png')]
    
    if len(valid_files) != SEQUENCE_LENGTH:
        print(f"Error: Expected {SEQUENCE_LENGTH} frames, found {len(valid_files)}.")
        return

    for file in valid_files:
        path = os.path.join(processed_folder, file)
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        frames.append(img_tensor)
    
    input_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
    
    print("Loading model...")
    model = VisionSnare(lstm_hidden_dim=512).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}!")
        return
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Running prediction...")
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()
        
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else 1 - prob
        
    print("\n" + "="*40)
    print(f"VIDEO: {os.path.basename(processed_folder).replace('_processed', '')}")
    print(f"PREDICTION: {prediction}")
    print(f"CONFIDENCE: {confidence*100:.2f}%")
    print(f"RAW SCORE:  {prob:.4f} (0=Real, 1=Fake)")
    print("="*40 + "\n")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    if not os.path.exists(VIDEO_INPUT_FOLDER):
        os.makedirs(VIDEO_INPUT_FOLDER)
        print(f"Created folder '{VIDEO_INPUT_FOLDER}'. Please put a video file inside it and run again.")
        exit()
        
    video_files = [f for f in os.listdir(VIDEO_INPUT_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in '{VIDEO_INPUT_FOLDER}'.")
    else:
        video_name = video_files[0]
        video_path = os.path.join(VIDEO_INPUT_FOLDER, video_name)
        processed_dir_name = os.path.splitext(video_name)[0] + "_processed"
        processed_path = os.path.join(VIDEO_INPUT_FOLDER, processed_dir_name)
        
        if os.path.exists(processed_path) and len(os.listdir(processed_path)) == SEQUENCE_LENGTH:
            print(f"Found existing processed frames in {processed_path}. Skipping preprocessing.")
            success = True
        else:
            success = preprocess_video_pipeline(video_path, processed_path)
        
        if success:
            predict(processed_path)