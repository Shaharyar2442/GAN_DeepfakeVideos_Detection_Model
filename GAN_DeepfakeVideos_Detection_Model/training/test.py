import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import FakeAVCelebSequenceDataset, get_default_transform
from models.visionsnare import VisionSnare


#TEST_DATA_ROOT = r"/home/moazzam/Documents/training_model/GAN_DeepfakeVideos_Detection_Model/Processed_FakeAVCeleb"
#TEST_DATA_ROOT = r"/home/moazzam/Documents/FAce_Forensics/preprocessed"


#TEST_DATA_ROOT = r"E:\test_data"
#TEST_DATA_ROOT = r"E:\Processed_FakeAVCeleb\test_data"
TEST_DATA_ROOT = r"E:\DeepfakeTimit_Test_Data"

CHECKPOINT_PATH = os.path.join("checkpoints", "model_best.pth")

# Hyperparameters 
BATCH_SIZE = 8
SEQUENCE_LENGTH = 12
LSTM_HIDDEN_DIM = 512

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Testing Configuration ---")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Test Data Root: {TEST_DATA_ROOT}")
print(f"Checkpoint Path: {CHECKPOINT_PATH}")
print("-----------------------------\n")

# Helper Functions

def calculate_predictions(logits):
    # Converts logits to binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return preds

def test(model, test_loader, criterion, device):
    # Runs the model on the test set and collects predictions
    model.eval() # Setting model to evaluation mode
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # No gradient calculation needed for testing
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for sequences, labels in progress_bar:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(sequences)
            
            # Calculate loss
            loss = criterion(logits.squeeze(1), labels.float())
            running_loss += loss.item() * sequences.size(0)
            
            # Get predictions
            preds = calculate_predictions(logits)
            
            # Store predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            
    total_loss = running_loss / len(test_loader.dataset)
    
    return total_loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)

# Main Test Function
def main():
    # Check paths
    if not os.path.exists(TEST_DATA_ROOT):
        print(f"Error: Test data directory not found at {TEST_DATA_ROOT}")
        return
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Best model checkpoint not found at {CHECKPOINT_PATH}. Have you trained yet?")
        return

    # Data Loading
    print("Initializing Test Dataset...")
    test_dataset = FakeAVCelebSequenceDataset(
        root_dir=TEST_DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        transform=get_default_transform()
    )
    
    if len(test_dataset) == 0:
        print("Error: No valid test samples found.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test dataset loaded. Total samples: {len(test_dataset)}")

    #  Model Setup
    print("Initializing VisionSnare Model...")
    model = VisionSnare(lstm_hidden_dim=LSTM_HIDDEN_DIM)
    model = model.to(device)
    
    #  Load Best Checkpoint
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully.")

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Run Testing
    print("\n--- Starting Evaluation ---")
    test_loss, preds, labels, probs = test(model, test_loader, criterion, device)
    
    #  Calculate and Print Metrics
    print("\n--- Final Test Results ---")
    accuracy = accuracy_score(labels, preds)
    auc_score = roc_auc_score(labels, probs) # Calculate AUC
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Final Test AUC: {auc_score:.4f}")
    
    print("\nClassification Report:")
    # target_names maps 0 to 'Real' and 1 to 'Fake'
    print(classification_report(labels, preds, target_names=['Real', 'Fake'], digits=4))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    print("(Top-Left: TN, Top-Right: FP, Bottom-Left: FN, Bottom-Right: TP)")

if __name__ == '__main__':
    main()