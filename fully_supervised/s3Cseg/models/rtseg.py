# -------------------------------------------------------------------------------------------------
# Implementation of the architecture proposed in
# RT-Seg: A Real-Time Semantic Segmentation Network for Side-Scan Sonar Images
# by Wang et al., 2019
# https://www.mdpi.com/1424-8220/19/9/1985
# -------------------------------------------------------------------------------------------------


import torch
import torch.nn as nn
from models.registry import arch_entrypoints


class ConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
     
   
class DepthSepConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                            groups=in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.bn1(self.dw_conv(x))
        x = self.relu(x)
        x = self.bn2(self.pw_conv(x))
        return x
    
        
class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, expansion_factor):
        super().__init__()
        hidden_dim = in_channels*expansion_factor
        self.expansion_layer = ConvBnRelu(in_channels, hidden_dim, 1, 1, 0)
        self.left_branch = DepthSepConv(hidden_dim, in_channels, 3, 1, 1)
        self.right_branch = DepthSepConv(hidden_dim, in_channels, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, return_indices=True)
    
    def forward(self, x):
        x = self.expansion_layer(x)
        x_trace = torch.cat([self.left_branch(x), self.right_branch(x)], dim=1)
        x, idx = self.pool(x_trace)
        return x, x_trace, idx
        
        
class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.first_layer = ConvBnRelu(2*in_channels, out_channels, 1, 1, 0)
        self.second_layer = ConvBnRelu(out_channels, out_channels, 3, 1, 1)
        
    def forward(self, x, x_trace, indices):
        x = self.unpool(x, indices)
        x = torch.cat([x, x_trace], dim=1)
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x
    
    
class RTSegNet(nn.Module):
    
    def __init__(self, in_channels, num_classes, expansion_factor=6):
        super().__init__()
        self.init_block = nn.Sequential(
            ConvBnRelu(in_channels, 16, 3, 1, 1),
            nn.MaxPool2d(2)
        )
        self.encoder = nn.ModuleList([
            EncoderBlock(16, expansion_factor),
            EncoderBlock(32, expansion_factor),
            EncoderBlock(64, expansion_factor),
            EncoderBlock(128, expansion_factor)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, num_classes)
        ])
        self.upsample = nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        indices = []
        x_trace = []
        x = self.init_block(x)
        for block in self.encoder:
            x, x_, idx = block(x)
            indices.append(idx)
            x_trace.append(x_)
        for block in self.decoder:
            x = block(x, x_trace.pop(), indices.pop())
        x = self.upsample(x)
        return x


@arch_entrypoints.register('rtseg')
def build_model(config):
    return RTSegNet(
        in_channels=config.DATA.IN_CHANS,
        num_classes=config.DATA.NUM_CLASSES,
        expansion_factor=config.MODEL.EXPANSION_FACTOR,
    )
