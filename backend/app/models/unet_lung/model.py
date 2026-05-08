"""UNet model implementation"""
import torch
import torch.nn as nn

class UNetLung(nn.Module):
    """UNet model for lung segmentation"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(UNetLung, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO: Implement UNet architecture
    
    def forward(self, x):
        """Forward pass"""
        # TODO: Implement forward pass
        pass
