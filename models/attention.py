import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-attention module for GANs (SAGAN-style).
    Allows the model to focus on long-range dependencies in the image.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: input feature map (B, C, H, W)
        Returns:
            out: self attention value + input feature (B, C, H, W)
        """
        batch_size, C, height, width = x.size()
        
        # Project to query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, HW, C//8)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C//8, HW)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, HW)
        
        # Compute attention scores
        attention = torch.bmm(proj_query, proj_key)  # (B, HW, HW)
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch_size, C, height, width)
        
        # Residual connection with learnable scale
        out = self.gamma * out + x
        
        return out

