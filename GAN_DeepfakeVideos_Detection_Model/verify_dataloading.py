import torch
from torch.utils.data import DataLoader
from data.dataset import FakeAVCelebSequenceDataset, get_default_transform
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# UPDATE THIS PATH to your actual data location
data_root = r"E:\Processed_FakeAVCeleb" 
batch_size = 4

# --- Helper to show a tensor image ---
def show_frame(tensor_img, title=None):
    # Convert from Tensor [C, H, W] to numpy [H, W, C] for plotting
    img = tensor_img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

# --- Main Test Loop ---
if __name__ == '__main__':
    print("Initializing Dataset...")
    # 1. Create the Dataset
    dataset = FakeAVCelebSequenceDataset(
        root_dir=data_root,
        sequence_length=12, # Based on your data
        transform=get_default_transform()
    )

    # Basic checks
    print(f"\nDataset Length: {len(dataset)}")
    if len(dataset) == 0:
        print("ERROR: No data found. Check your path.")
        exit()

    # 2. Create the DataLoader
    # This handles batching and shuffling
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("DataLoader created.")

    # 3. Grab one batch to test
    print("\nLoading one batch...")
    # iter(loader).next() gets the first batch
    train_features, train_labels = next(iter(train_loader))

    # --- Verify Shapes ---
    print(f"\nBatch Shape Verification:")
    # Expected: [Batch_Size, Seq_Len, Channels, Height, Width]
    # E.g., [4, 12, 3, 256, 256]
    print(f"Feature Batch Shape: {train_features.shape}") 
    # Expected: [Batch_Size] -> e.g., [4]
    print(f"Label Batch Shape:   {train_labels.shape}")

    assert train_features.shape == (batch_size, 12, 3, 256, 256), "Bad feature shape!"
    assert train_labels.shape == (batch_size,), "Bad label shape!"
    print("✅ Shapes are correct!")
    print(f"Labels in this batch: {train_labels}")

    # --- Visual Verification ---
    # Let's plot the first frame of the first video in the batch
    print("\nVisualizing first frame of the first sample in batch...")
    first_video_sequence = train_features[0] # Shape [12, 3, 256, 256]
    first_frame = first_video_sequence[0]   # Shape [3, 256, 256]
    label = train_labels[0].item()
    label_str = "Fake" if label == 1 else "Real"
    
    plt.figure(figsize=(4,4))
    show_frame(first_frame, title=f"Label: {label} ({label_str})")
    plt.show()
    
    print("\nVerification complete. Data pipeline is ready.")