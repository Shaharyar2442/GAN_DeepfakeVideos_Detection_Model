import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialNPR(nn.Module):
    
    #This class calculates the 2x2 Spatial Neighboring Pixel Relationship (NPR) map for an input frame.
    #The output has the same spatial dimensions (H, W) as the input.
    
    def __init__(self):
        super(SpatialNPR, self).__init__()

    def forward(self, x): # x= torch.Tensor (Input tensor of shape [Batch*SeqLen, 3, H, W])
        
       # Returns:  torch.Tensor: NPR map of shape [Batch*SeqLen, 3, H, W]
        
        # Getting current height and width
        h, w = x.shape[2], x.shape[3]

        # Calculating necessary padding to make H and W divisible by 2 so that 2x2 patches can be formed
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        
        # Padding the frame.
        img_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Unfolding the image into non-overlapping 2x2 patches
        # Shape becomes: [Batch, 3, H_pad/2, W_pad/2, 2, 2]
        patches = img_pad.unfold(2, 2, 2).unfold(3, 2, 2)
        
        # Main spatial NPR calculation:
        # Subtract the top-left pixel (index 0,0) of each patch from all pixels in that patch.
        # Then take the absolute value.
        top_left_pixels = patches[:, :, :, :, 0:1, 0:1]
        npr_patches = torch.abs(patches - top_left_pixels)
        
        # Reshaping the patches back into the original image dimensions
        #  Permuting dimensions to order them for reshaping: [B, C, H/2, 2, W/2, 2]
        #  Reshaping back to [B, C, H_pad, W_pad]
        h_pad_new = img_pad.shape[2]
        w_pad_new = img_pad.shape[3]
        npr_map = npr_patches.permute(0, 1, 2, 4, 3, 5).reshape(x.shape[0], x.shape[1], h_pad_new, w_pad_new)
        
        # Cropping off any padding that was added to return to original dimensions
        if pad_h > 0 or pad_w > 0:
            npr_map = npr_map[:, :, :h, :w]
            
        return npr_map

# --- Verification Code ---
if __name__ == '__main__':
    # Testing with a dummy batch of 2 RGB frames
    npr_layer = SpatialNPR()
    dummy_input = torch.randn(2, 3, 256, 256)
    output = npr_layer(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print(f"Top-left 2x2 block of output (channel 0):\n{output[0,0,0:2,0:2]}")
    assert output.shape == dummy_input.shape
    print(" SpatialNPR Module verification passed!")