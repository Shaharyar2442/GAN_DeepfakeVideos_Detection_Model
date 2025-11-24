import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import natsort # Important for sorting frame names correctly

class FakeAVCelebSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading sequences of frames from
    the processed FakeAVCeleb directory structure.
    """

    def __init__(self, root_dir, sequence_length=12, transform=None):
        """
        Args:
            root_dir (str or Path): Root directory containing '0_real' and '1_fake' folders
                                    (e.g., "E:/Processed_FakeAVCeleb").
            sequence_length (int): Expected number of frames per video folder.
            transform (callable, optional): Optional transform to be applied to EACH frame.
        """
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Define class mappings
        self.classes = ['0_real', '1_fake']
        self.class_to_idx = {'0_real': 0, '1_fake': 1}
        
        # This list will hold tuples of (path_to_video_folder, label_index)
        self.samples = []
        
        print(f"Scanning data directory: {self.root_dir} ...")
        self._scan_directory()
        print(f"Found {len(self.samples)} valid video sequences.")


    def _scan_directory(self):
        """Helper to scan folders and populate self.samples"""
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory '{class_dir}' not found.")
                continue
            
            label = self.class_to_idx[class_name]
            
            # Iterate through each video folder inside '0_real' or '1_fake'
            # (e.g., E:/Processed_FakeAVCeleb/0_real/id00076_00109_1)
            for video_folder in class_dir.iterdir():
                if video_folder.is_dir():
                    # Quick check: does it have enough frames?
                    # We look for .png files
                    frame_count = len(list(video_folder.glob("*.png")))
                    
                    if frame_count == self.sequence_length:
                        # Add to our list of valid samples
                        self.samples.append((video_folder, label))
                    else:
                        # This should ideally not happen based on your preprocessing,
                        # but it's good defensive coding.
                        print(f"Warning: Skipping {video_folder.name}, found {frame_count} frames, expected {self.sequence_length}.")


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)


    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        
        Returns:
            tuple: (sequence_tensor, label)
            - sequence_tensor shape: [sequence_length, 3, H, W]
            - label: int (0 or 1)
        """
        video_folder_path, label = self.samples[idx]
        
        # Get all frame files and sort them naturally (0000, 0001, 0002...)
        frame_files = natsort.natsorted(list(video_folder_path.glob("*.png")))
        
        # Double check length (safety first)
        if len(frame_files) != self.sequence_length:
             raise RuntimeError(f"Error loading {video_folder_path.name}: Expected {self.sequence_length} frames, found {len(frame_files)}")
             
        frames_list = []
        
        # Load each image in the sequence
        for frame_path in frame_files:
            # Use PIL to load. convert('RGB') ensures 3 channels.
            img = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                # Apply transforms (like converting to tensor, normalizing)
                img_tensor = self.transform(img)
            else:
                # Fallback: just convert to tensor if no transform provided
                img_tensor = transforms.ToTensor()(img)
                
            frames_list.append(img_tensor)
            
        # Stack the list of 12 tensors into one big tensor
        # Result shape: [12, 3, 256, 256]
        sequence_tensor = torch.stack(frames_list)
        
        return sequence_tensor, label

# --- Default transforms for this dataset ---
# This is what converts PIL images to PyTorch Tensors in the [0, 1] range.
# We can add normalization here later if needed.
def get_default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])