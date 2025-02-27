import torch
from torch import nn as nn
from torch.nn import functional as F

class StemConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels//2, kernel_size = 3, padding =1,stride = (2,2,2))
        self.norm1 = nn.GroupNorm(1, out_channels//2, eps=1e-5, affine=True)     
        self.non_linearity = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels//2, out_channels, kernel_size = 3, padding=1, stride = 1)
        self.norm2 = nn.GroupNorm(1, out_channels, eps=1e-5, affine=True)   

        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding=1, stride = (1,1,1))  

    def forward(self, x):
        x = self.non_linearity(self.norm1(self.conv1(x)))
        x = self.non_linearity(self.norm2(self.conv2(x)))
        # out = self.non_linearity(self.norm3(self.conv3(x)))
        out = self.conv3(x)
        # print("++++++++++++++++ ", out.shape) #(14, 40, 40)

        return out


class ExpandingConv(nn.Module):
    def __init__(self, in_channels, scale_factor=4):
        super(ExpandingConv, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,  # Keep the same number of channels
            kernel_size=(2, 2, 2),  # Upscaling by factor of 2
            stride=(2, 2, 2),
            padding=(0, 0, 0),
            output_padding=(0, 0, 0)
        )
        
        # Second stage: Upsize (14, 40, 40) to (14, 160, 160)
        self.deconv2 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,  # Keep the same number of channels
            kernel_size=(3, 3, 3),  # Upscaling by factor of 4
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            output_padding=(0, 0, 0)
        )
        
        # Optional activation
        self.activation = nn.ReLU(inplace=True)
        self.norm1 = nn.GroupNorm(1, in_channels) 
        self.norm2 = nn.GroupNorm(1, in_channels) 

    def forward(self, x):
        x = self.activation(self.deconv1(self.norm1(x)))  # Stage 1
        x = self.deconv2(self.norm2(x))  # Stage 2
        return x













