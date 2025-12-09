import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.npr import SpatialNPR
from models.backbone import get_lightweight_backbone
from models.attention import FrameAttention

class VisionSnare(nn.Module):
    """
    The full VisionSnare architecture for Video Deepfake Detection.
    Pipeline:
    1. Spatial NPR -> 2. Temporal NPR Diff -> 3. CNN Feature Extractor
    -> 4. Attention Mechanism -> 5. LSTM -> 6. Classifier Head
    """
    def __init__(self, lstm_hidden_dim=512, num_lstm_layers=1):
        super(VisionSnare, self).__init__()
        
        # Spatial NPR Module
        self.spatial_npr = SpatialNPR()
        
        # CNN Feature Extractor
        # Getting the custom lightweight backbone (using 'resnet50' style blocks)
        # This returns the model and its output feature dimension (512)
        self.cnn, self.cnn_feature_dim = get_lightweight_backbone(architecture='resnet50')
        print(f"VisionSnare initialized with CNN feature dim: {self.cnn_feature_dim}")
        
        # The Attention mechanism operates on the CNN's output dimension
        self.attention = FrameAttention(feature_dim=self.cnn_feature_dim)
        
        # LSTM: Takes CNN features, outputs lstm_hidden_dim vectors
        # batch_first=True means input is [Batch, SeqLen, Features]
        self.lstm = nn.LSTM(input_size=self.cnn_feature_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        
        # Classifier Head: Maps final LSTM state to a single logit (for binary class)
        self.classifier = nn.Linear(lstm_hidden_dim, 1)
        

    def forward(self, x):
        
        #Args:x (torch.Tensor): Input batch of sequences.
        #Returns:torch.Tensor: Final logits of shape [Batch, 1]
        
        batch_size, seq_len, C, H, W = x.shape
        
        
        # Step 1: Feature Engineering (Calculating Spatial & Temporal NPR)
        
        #  Calculating Spatial NPR for ALL frames at once for efficiency.
        # Reshaping input to [B*S, 3, H, W] to feed into the spatial module.
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        spatial_npr_maps = self.spatial_npr(x_reshaped)
        # Reshaping back to sequence format: [B, S, 3, H, W]
        spatial_npr_seq = spatial_npr_maps.view(batch_size, seq_len, C, H, W)
        
        # Step 2: Calculate Temporal NPR (Frame Differencing)
        # We take frames t=1 to end, and subtract frames t=0 to end-1.
        # Resulting sequence length will be seq_len - 1 (i.e. 11 frames).
        frames_t1 = spatial_npr_seq[:, 1:, :, :, :] # Frames [1, 2, ..., 11]
        frames_t0 = spatial_npr_seq[:, :-1, :, :, :] # Frames [0, 1, ..., 10]
        temporal_npr_seq = torch.abs(frames_t1 - frames_t0)
        
        # Step 3: Creating  Final 6-Channel Input
        #Using the corresponding spatial frames (t=1 to end) to match temporal.
        spatial_input_seq = spatial_npr_seq[:, 1:, :, :, :]
        
        # Concatenating along the channel dimension (dim=2)
        # Shape becomes: [Batch, SeqLen-1, 6, H, W]
        final_input_seq = torch.cat([spatial_input_seq, temporal_npr_seq], dim=2)
        
        new_seq_len = seq_len - 1
        
        # Step 4: CNN Feature Extraction
        
        # Flattenning sequence dimension to feed into CNN: [B*(S-1), 6, H, W]
        cnn_input = final_input_seq.view(batch_size * new_seq_len, 6, H, W)
        
        # Extracting features. Output shape: [B*(S-1), cnn_feature_dim]
        cnn_features_flat = self.cnn(cnn_input)
        
        # Reshaping back to sequence format: [B, S-1, cnn_feature_dim]
        cnn_features_seq = cnn_features_flat.view(batch_size, new_seq_len, self.cnn_feature_dim)
        
        # Step 5: Attention Mechanism
        
        # Applying attention to the sequence of feature vectors
        # weighted_features shape: [B, S-1, cnn_feature_dim]
        weighted_features, attention_weights = self.attention(cnn_features_seq)
        
        # Step 6: Temporal Aggregation (LSTM)
        
        # Passing the weighted sequence through the LSTM
        # lstm_out shape: [B, S-1, hidden_dim] (output at every step)
        # hidden shape: (h_n, c_n) -> h_n is [num_layers, B, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(weighted_features)
        
        # We want the final hidden state of the last layer.
        # It represents the summary of the entire sequence.
        # Shape: [Batch, hidden_dim]
        final_hidden_state = hidden[-1]
        
        # Step 7: Classifier Head
        
        # Mapping the final hidden state to a single logit score
        # Shape: [Batch, 1]
        logits = self.classifier(final_hidden_state) 
        
        return logits

# Verification Block
if __name__ == '__main__':
    
    print("Initializing full VisionSnare model...")
    # Create the model with a 512-dim LSTM hidden state
    model = VisionSnare(lstm_hidden_dim=512)
    
    # Dummy input: Batch=2, SeqLen=12, Channels=3, H=256, W=256
    dummy_input = torch.randn(2, 12, 3, 256, 256)
    print(f"Model input shape: {dummy_input.shape}")
    
    # Running the forward pass
    print("Running forward pass...")
    with torch.no_grad():
        logits = model(dummy_input)
    
    print(f"\nFinal Output (Logits) shape: {logits.shape}")
    print(f"Final Output values:\n{logits}")
    
    # Verification criteria
    # 1. Output shape must be [BatchSize, 1]
    assert logits.shape == (2, 1), "Error: Final output shape is incorrect!"
    
    # 2. The sequence length should have been reduced by 1 internally
    
    print("\n GRAND SUCCESS: Full VisionSnare model is assembled and working!")