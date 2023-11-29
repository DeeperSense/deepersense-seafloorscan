# -------------------------------------------------------------------------------------------------
# Implementation of the architecture proposed in
# ECNet: Efficient Convolutional Networks for Side Scan Sonar Image Segmentation
# by Wu et al., 2019
# https://www.mdpi.com/1424-8220/19/9/2009
# -------------------------------------------------------------------------------------------------


import torch
import torch.nn as nn
from models.registry import arch_entrypoints


class BnReluConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x 


class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_conv = nn.Sequential(
            BnReluConv(in_channels, in_channels, 3, 1, 1),
            BnReluConv(in_channels, in_channels, 3, 1, 1)
        )
        self.conv = BnReluConv(in_channels, out_channels, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        
    def forward(self, x):
        x = self.res_conv(x) + x
        x = self.conv(x)
        x, idx = self.pool(x)
        return x, idx


class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.conv = BnReluConv(in_channels, out_channels, 3, 1, 1)
        
    def forward(self, x, x_trace, indices):
        if x_trace is not None:
            x = x + x_trace
        x = self.unpool(x, indices)
        x = self.conv(x)
        return x

    
class SideOutput(nn.Module):
    
    def __init__(self, in_channels, out_channels, out_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.up = nn.Upsample(size=out_size, mode='bilinear')
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        x = self.act(x)
        return x
  

class ECNet(nn.Module):
    
    def __init__(self, img_size, in_channels, num_classes):
        super().__init__()
        self.encoder = nn.ModuleList([
            EncoderBlock(in_channels, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32)
        ])
        self.side = nn.ModuleList([
            SideOutput(64, num_classes, img_size),
            SideOutput(128, num_classes, img_size),
            SideOutput(256, num_classes, img_size)
        ])
        self.out = nn.Sequential(
            nn.Conv2d(32, num_classes, 1, 1, 0),
            nn.Sigmoid()
        )
        self.img_size = img_size
        self.num_classes = num_classes
        
    def forward(self, x):
        indices = []
        x_trace = []
        x_ = torch.zeros((x.shape[0], self.num_classes, self.img_size, self.img_size),
                device=x.device)
        for i, block in enumerate(self.encoder):
            x, idx = block(x)
            indices.append(idx)
            if i < 3:
                x_trace.append(x)
                x_ += self.side[i](x)
            else:
                x_trace.append(None)
        for block in self.decoder:
            x = block(x, x_trace.pop(), indices.pop())
        x = self.out(x)
        x = (x + x_)/4
        return x


@arch_entrypoints.register('ecnet')
def build_model(config):
    return ECNet(
        img_size=config.DATA.IMAGE_SIZE,
        in_channels=config.DATA.IN_CHANS,
        num_classes=config.DATA.NUM_CLASSES,
    )
