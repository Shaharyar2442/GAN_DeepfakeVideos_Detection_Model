"""
VisionSnare — Kaggle/Colab Testing Script
==========================================
This script evaluates the trained VisionSnare model on any preprocessed dataset.
Run this AFTER kaggle_preprocessing.py has created the processed directory.

It loads the model architecture, loads the trained weights from model_best.pth,
runs inference on the processed dataset, and outputs:
  - Accuracy, AUC, Loss
  - Classification Report (Precision, Recall, F1)
  - Confusion Matrix
  - All results saved to a .txt file

Usage:
  1. Edit TEST_DATA_ROOT, CHECKPOINT_PATH, and DATASET_NAME below
  2. Run: python testing_kaggle.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from datetime import datetime

# Add parent/current directory to path so we can import our model and dataset classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import FakeAVCelebSequenceDataset, get_default_transform
from models.visionsnare import VisionSnare

# ============================================================
# >>>  EDIT THIS SECTION FOR YOUR DATASET  <<<
# ============================================================

# --- Celeb-DF v2 (Default) ---
#TEST_DATA_ROOT = "/kaggle/input/datasets/shaharyarrizwan/processed-celebdf/Processed_CelebDF"
TEST_DATA_ROOT="/kaggle/input/datasets/shaharyarrizwan/processed-df40/kaggle/working/Processed_df40"
DATASET_NAME = "df40"

# --- FaceForensics++ (uncomment to use instead) ---
# TEST_DATA_ROOT = "/kaggle/working/Processed_FaceForensics"
# DATASET_NAME = "FaceForensics"

# --- DFDC (uncomment to use instead) ---
# TEST_DATA_ROOT = "/kaggle/working/Processed_DFDC"
# DATASET_NAME = "DFDC"

# --- DeepfakeTIMIT (uncomment to use instead) ---
# TEST_DATA_ROOT = "/kaggle/working/Processed_DeepfakeTIMIT"
# DATASET_NAME = "DeepfakeTIMIT"

# --- Custom (uncomment and edit) ---
# TEST_DATA_ROOT = "/kaggle/working/Processed_Custom"
# DATASET_NAME = "CustomDataset"

# ============================================================
# >>>  MODEL CONFIGURATION (should match training settings)  <<<
# ============================================================
# Update this path to match where your model_best.pth is on Kaggle
CHECKPOINT_PATH = "/kaggle/input/datasets/shaharyarrizwan/visionsnare-project/visionsnare-project/checkpoints/model_best.pth"
BATCH_SIZE = 16           # Larger batch for inference (no gradients = less GPU memory needed)
SEQUENCE_LENGTH = 12
LSTM_HIDDEN_DIM = 512
NUM_WORKERS = 2           # Parallel data loading workers (CPU threads loading data while GPU runs)
# ============================================================


# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU optimizations
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms for fixed input sizes


def calculate_predictions(logits):
    """Converts logits to binary predictions using sigmoid + 0.5 threshold."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return preds


def test(model, test_loader, criterion, device):
    """Runs the model on the test set and collects all predictions.
    GPU-optimized with mixed precision (float16) inference for faster computation."""
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    # Use mixed precision for faster GPU inference
    # autocast runs forward pass in float16 where safe, reducing memory and increasing throughput
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for sequences, labels in progress_bar:
            # non_blocking=True allows async CPU→GPU transfer (works with pin_memory)
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(sequences)
                loss = criterion(logits.squeeze(1), labels.float())

            running_loss += loss.item() * sequences.size(0)

            # Get predictions
            preds = calculate_predictions(logits)

            # Store for metrics calculation
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())

    total_loss = running_loss / len(test_loader.dataset)

    return total_loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    # Print configuration
    print("=" * 60)
    print("VisionSnare — Kaggle/Colab Testing Script")
    print("=" * 60)
    print(f"Device:          {device}")
    if device.type == 'cuda':
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Dataset:         {DATASET_NAME}")
    print(f"Test Data Root:  {TEST_DATA_ROOT}")
    print(f"Checkpoint Path: {CHECKPOINT_PATH}")
    print("=" * 60)

    # Check paths
    if not os.path.exists(TEST_DATA_ROOT):
        print(f"\nError: Test data directory not found at: {TEST_DATA_ROOT}")
        print("Make sure you have run kaggle_preprocessing.py first!")
        return
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nError: Model checkpoint not found at: {CHECKPOINT_PATH}")
        print("Make sure model_best.pth is in the checkpoints/ folder.")
        return

    # Data Loading
    print("\nInitializing Test Dataset...")
    test_dataset = FakeAVCelebSequenceDataset(
        root_dir=TEST_DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        transform=get_default_transform()
    )

    if len(test_dataset) == 0:
        print("Error: No valid test samples found.")
        print("Make sure the processed directory has 0_real/ and 1_fake/ folders")
        print("with video subfolders containing exactly 12 PNG frames each.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,   # Parallel data loading (CPU loads next batch while GPU processes current)
        pin_memory=True if device.type == 'cuda' else False,  # Faster CPU→GPU memory transfer
        persistent_workers=True if NUM_WORKERS > 0 else False  # Keep workers alive between batches
    )
    print(f"Test dataset loaded. Total samples: {len(test_dataset)}")

    # Model Setup
    print("\nInitializing VisionSnare Model...")
    model = VisionSnare(lstm_hidden_dim=LSTM_HIDDEN_DIM)
    model = model.to(device)

    # Load Best Checkpoint (with weights_only=True for security)
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully.")

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Run Testing
    print("\n--- Starting Evaluation ---")
    test_loss, preds, labels, probs = test(model, test_loader, criterion, device)

    # Calculate Metrics
    accuracy = accuracy_score(labels, preds)

    # AUC requires both classes to be present
    try:
        auc_score = roc_auc_score(labels, probs)
        auc_str = f"{auc_score:.4f}"
    except ValueError:
        auc_score = None
        auc_str = "N/A (only one class present)"

    report = classification_report(labels, preds, target_names=['Real', 'Fake'], digits=4)
    cm = confusion_matrix(labels, preds)

    # Build results string
    results = []
    results.append("=" * 60)
    results.append(f"VisionSnare Test Results — {DATASET_NAME}")
    results.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results.append("=" * 60)
    results.append(f"Device: {device}")
    if device.type == 'cuda':
        results.append(f"GPU: {torch.cuda.get_device_name(0)}")
    results.append(f"Test Data: {TEST_DATA_ROOT}")
    results.append(f"Total Samples: {len(test_dataset)}")
    results.append(f"Checkpoint: {CHECKPOINT_PATH}")
    results.append("-" * 60)
    results.append(f"Final Test Loss:     {test_loss:.4f}")
    results.append(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    results.append(f"Final Test AUC:      {auc_str}")
    results.append("")
    results.append("Classification Report:")
    results.append(report)
    results.append("Confusion Matrix:")
    results.append(str(cm))
    results.append("(Top-Left: TN, Top-Right: FP, Bottom-Left: FN, Bottom-Right: TP)")
    results.append("=" * 60)

    # Print results to console
    results_text = "\n".join(results)
    print(f"\n{results_text}")

    # Save results to file
    output_filename = f"test_results_{DATASET_NAME}.txt"
    with open(output_filename, "w") as f:
        f.write(results_text)
    print(f"\nResults saved to: {output_filename}")


if __name__ == '__main__':
    main()
