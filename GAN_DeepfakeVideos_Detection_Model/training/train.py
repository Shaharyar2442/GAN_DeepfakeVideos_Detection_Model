import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # For progress bars
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import FakeAVCelebSequenceDataset, get_default_transform
from models.visionsnare import VisionSnare

DATA_ROOT = r"E:\Processed_FakeAVCeleb"  
CHECKPOINT_DIR = "checkpoints"           #Creating a directory named checkpoints in the current working directory

# Hyperparameters
BATCH_SIZE = 8         
LEARNING_RATE = 1e-4   
NUM_EPOCHS = 20       
SEQUENCE_LENGTH = 12   
LSTM_HIDDEN_DIM = 512  
VAL_SPLIT_RATIO = 0.2  # 20% of data for validation

# Setting up device and directories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training Configuration ---")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Data Root: {DATA_ROOT}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Validation Split: {VAL_SPLIT_RATIO*100}%")
print("------------------------------\n")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Helper Functions

def save_checkpoint(state, filename="checkpoint_latest.pth"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    print(f"-> Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filename="checkpoint_latest.pth"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        print(f"-> Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Tracking best VALIDATION accuracy
        best_val_acc = checkpoint.get('best_val_acc', 0.0) 
        print(f"-> Checkpoint loaded. Resuming from Epoch {start_epoch}. Best Val Acc: {best_val_acc:.4f}")
        return start_epoch, best_val_acc
    else:
        print(f"-> No checkpoint found at {filepath}. Starting fresh.")
        return 0, 0.0 

def calculate_accuracy(logits, labels): # Function to calculate accuracy by comparing predicted labels with true labels
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    correct = (preds.squeeze() == labels.float()).sum().item()
    return correct

#  Validation Loop Function
def validate(model, val_loader, criterion, device):
    #Runs the model on the validation set
    model.eval() # Set model to evaluation mode 
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # No gradient calculation needed for validation (saves memory and time)
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Validation", unit="batch", leave=False):
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = criterion(logits.squeeze(1), labels.float())

            running_loss += loss.item() * sequences.size(0)
            running_corrects += calculate_accuracy(logits, labels)
            total_samples += sequences.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

# Main Training Function
def train():
    # 1. Data Loading & Splitting
    print("Initializing full dataset...")
    full_dataset = FakeAVCelebSequenceDataset(
        root_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        transform=get_default_transform()
    )
    
    # Calculating split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT_RATIO)
    train_size = total_size - val_size
    
    print(f"Total samples: {total_size}. Splitting into Train: {train_size}, Val: {val_size}")
    
    # Performing random split
    # Setting a random seed generator for reproducibility
    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Creating DataLoaders
    # Shuffling train data, not shuffling validation data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Model Setup
    print("Initializing VisionSnare Model...")
    model = VisionSnare(lstm_hidden_dim=LSTM_HIDDEN_DIM)
    model = model.to(device) 

    # 3. Loss ftn and Optimizer
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss used for binary classification 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Resuming training from Checkpoint
    start_epoch, best_val_acc = load_checkpoint(model, optimizer)

    # 5. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, NUM_EPOCHS):
        #  Training Phase 
        model.train() # Setting  model to training mode
        
        train_loss = 0.0
        train_corrects = 0
        train_samples = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")
        
        for batch_idx, (sequences, labels) in progress_bar: # Iterating through batches
            sequences = sequences.to(device) # Moving data to GPU if available
            labels = labels.to(device) # Moving data to GPU if available

            optimizer.zero_grad() # Clearing gradients
            logits = model(sequences) # Forward pass
            loss = criterion(logits.squeeze(1), labels.float()) # Computing loss
            loss.backward() # Backward pass
            optimizer.step() # Updating weights

            # Statistics
            batch_loss = loss.item()
            train_loss += batch_loss * sequences.size(0)
            train_corrects += calculate_accuracy(logits, labels)
            train_samples += sequences.size(0)

            progress_bar.set_postfix({'train_loss': f'{batch_loss:.4f}'})
            
        epoch_train_loss = train_loss / train_samples
        epoch_train_acc = train_corrects / train_samples

        # Validation Phase
        epoch_val_loss, epoch_val_acc = validate(model, val_loader, criterion, device)

        # Epoch Summary
        print(f"\nEnd of Epoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

        # Checkpointing
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc, # Save current best
        }
        # 1. Save latest checkpoint for resuming
        save_checkpoint(checkpoint_state, filename="checkpoint_latest.pth")
        
        # 2. Save best model based on VALIDATION accuracy
        if epoch_val_acc > best_val_acc:
            print(f"*** Found new best validation accuracy! ({best_val_acc:.4f} -> {epoch_val_acc:.4f}). Saving model_best.pth ***")
            best_val_acc = epoch_val_acc
            # Update state with new best before saving
            checkpoint_state['best_val_acc'] = best_val_acc
            save_checkpoint(checkpoint_state, filename="model_best.pth")
            
    print("\n--- Training Complete ---")

if __name__ == '__main__':
    train()