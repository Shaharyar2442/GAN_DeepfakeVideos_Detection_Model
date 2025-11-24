import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # For progress bars
import sys
import os
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import FakeAVCelebSequenceDataset, get_default_transform
from models.visionsnare import VisionSnare

# --- Configuration ---
# Path to your processed data
DATA_ROOT = r"E:\Processed_FakeAVCeleb" 
# Directory where checkpoints will be saved
CHECKPOINT_DIR = "checkpoints"

# Hyperparameters
BATCH_SIZE = 8         # Reduce if you get Out-of-Memory (OOM) errors on GPU
LEARNING_RATE = 1e-4   # A standard starting LR for Adam
NUM_EPOCHS = 1        # Total number of epochs to train
SEQUENCE_LENGTH = 12   # Fixed by your data processing
LSTM_HIDDEN_DIM = 512  # Match the CNN feature dimension for simplicity

# --- Setup Device and Directories ---
# Automatically detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training Configuration ---")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Data Root: {DATA_ROOT}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print("------------------------------\n")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Helper Functions ---

def save_checkpoint(state, filename="checkpoint_latest.pth"):
    """Saves the model and training state to a file."""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    print(f"-> Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filename="checkpoint_latest.pth"):
    """Loads training state from a checkpoint file."""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        print(f"-> Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0) # Default to 0 if not present
        print(f"-> Checkpoint loaded. Resuming from Epoch {start_epoch}.")
        return start_epoch, best_acc
    else:
        print(f"-> No checkpoint found at {filepath}. Starting fresh.")
        return 0, 0.0 # Start from epoch 0

def calculate_accuracy(logits, labels):
    """Calculates accuracy for a batch."""
    # Convert logits (scores) to probabilities using Sigmoid
    probs = torch.sigmoid(logits)
    # Threshold probabilities at 0.5 to get predicted class (0 or 1)
    preds = (probs > 0.5).float()
    # Count correct predictions
    correct = (preds.squeeze() == labels.float()).sum().item()
    return correct

# --- Main Training Function ---
def train():
    # 1. Data Loading
    print("Initializing Dataset & DataLoader...")
    dataset = FakeAVCelebSequenceDataset(
        root_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        transform=get_default_transform()
    )
    
    # For now, we use the whole dataset for training. 
    # Later we will split this into train/val sets.
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Data loaded. Total training samples: {len(dataset)}")

    # 2. Model Setup
    print("Initializing VisionSnare Model...")
    model = VisionSnare(lstm_hidden_dim=LSTM_HIDDEN_DIM)
    model = model.to(device) # Move model to CPU or GPU

    # 3. Loss and Optimizer
    # BCEWithLogitsLoss combines Sigmoid and BCE Loss, which is numerically more stable.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Resume from Checkpoint (if available)
    start_epoch, best_acc = load_checkpoint(model, optimizer)

    # 5. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train() # Set model to training mode
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Create a progress bar for the dataloader
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for batch_idx, (sequences, labels) in progress_bar:
            # Move data to device
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # Output shape: [Batch, 1]
            logits = model(sequences)
            
            # Calculate loss
            # Squeeze logits to [Batch] to match labels shape
            loss = criterion(logits.squeeze(1), labels.float())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            batch_loss = loss.item()
            running_loss += batch_loss * sequences.size(0) # Multiply by batch size for correct average
            
            batch_corrects = calculate_accuracy(logits, labels)
            running_corrects += batch_corrects
            total_samples += sequences.size(0)

            # Update progress bar with current batch loss
            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

        # --- Epoch End Statistics ---
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        print(f"End of Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        # --- Save Checkpoint ---
        # Save a general checkpoint after every epoch for resuming
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': epoch_acc
        }
        save_checkpoint(checkpoint_state, filename="checkpoint_latest.pth")
        
        # Save a separate checkpoint for the best model found so far
        # (For now, "best" is based on training accuracy, later we use validation)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print(f"New best training accuracy! Saving best model.")
            save_checkpoint(checkpoint_state, filename="model_best.pth")
            
    print("\n--- Training Complete ---")

# --- Run Script ---
if __name__ == '__main__':
    # This ensures the script runs when executed directly
    train()