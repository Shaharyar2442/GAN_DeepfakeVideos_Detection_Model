import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameAttention(nn.Module):
    """
    Implements a soft attention mechanism to learn importance weights for each frame
    in a sequence of feature vectors.
    """
    def __init__(self, feature_dim, hidden_dim=256):
        super(FrameAttention, self).__init__()
        
        # A small Feed-Forward Network (FFN) to calculate attention scores.
        # It maps the 2048-d feature vector to a single score.
        self.attention_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),                       # Non-linearity
            nn.Linear(hidden_dim, 1)         # Output a single scalar score per frame
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Sequence of features. Shape [Batch, SeqLen, FeatureDim]
                              e.g., [B, 11, 2048]
        Returns:
            weighted_features (torch.Tensor): Shape [Batch, SeqLen, FeatureDim]
            attention_weights (torch.Tensor): Shape [Batch, SeqLen, 1] (for visualization)
        """
        # 1. Calculate raw attention scores for each frame
        # Shape: [Batch, SeqLen, 1]
        scores = self.attention_layer(x)
        
        # 2. Apply Softmax across the sequence dimension (dim=1) to get weights that sum to 1
        # Shape: [Batch, SeqLen, 1]
        weights = F.softmax(scores, dim=1)
        
        # 3. Multiply original features by their corresponding weights
        # We use broadcasting here.
        # [B, S, F] * [B, S, 1] -> [B, S, F]
        weighted_features = x * weights
        
        return weighted_features, weights

# --- Small verification block ---
if __name__ == '__main__':
    # Dummy input: Batch=2, SeqLen=11, FeatureDim=512
    dummy_input = torch.randn(2, 11, 512)
    attention_module = FrameAttention(feature_dim=512)
    
    weighted_output, weights = attention_module(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Weighted output shape: {weighted_output.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # Verify weights sum to 1 along the sequence dimension
    sum_of_weights = torch.sum(weights[0], dim=0)
    print(f"Sum of weights for first sample (should be close to 1.0): {sum_of_weights.item()}")
    
    assert weighted_output.shape == dummy_input.shape
    assert torch.isclose(sum_of_weights, torch.tensor(1.0))
    print("✅ FrameAttention Module verification passed!")