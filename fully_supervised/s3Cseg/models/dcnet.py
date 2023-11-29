# -------------------------------------------------------------------------------------------------
# Implementation of the architecture proposed in
# DcNet: Dilated Convolutional Neural Networks for Side-Scan Sonar Image Semantic Segmentation
# by Xiaohong et al., 2021
# https://link.springer.com/article/10.1007/s11802-021-4668-5
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
                                groups = in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
    
class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, type):
        super().__init__()
        assert type in ('full_conv', 'mixed_conv', 'dws_conv'), "Invalid Type"
        if type == 'full_conv':
            self.conv = nn.Sequential(
                ConvBnRelu(in_channels, out_channels, 3, 1, 1),
                ConvBnRelu(out_channels, out_channels, 3, 1, 1),
            )
        elif type == 'mixed_conv':
            self.conv = nn.Sequential(
                ConvBnRelu(in_channels, out_channels, 3, 1, 1),
                DepthSepConv(out_channels, out_channels, 3, 1, 1),
                DepthSepConv(out_channels, out_channels, 3, 1, 1),
            )
        elif type == 'dws_conv':
            self.conv = nn.Sequential(
                DepthSepConv(in_channels, out_channels, 3, 1, 1),
                DepthSepConv(out_channels, out_channels, 3, 1, 1),
                DepthSepConv(out_channels, out_channels*2, 3, 1, 1),
            )
        self.pool = nn.MaxPool2d(2, return_indices=True)
        
    def forward(self, x):
        x = self.conv(x)
        x, indices = self.pool(x) 
        return x, indices
    

class DCBlock(nn.Module):

    def __init__(self, img_size):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, 1, 1)
        self.dilated_conv1 = nn.Conv2d(16, 8, 3, 1, 2, 2)
        self.dilated_conv2 = nn.Conv2d(32, 8, 3, 1, 2, 2)
        self.dilated_conv3 = nn.Conv2d(128, 8, 3, 1, 2, 2)
        self.dws_conv = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1, groups=24),
            nn.Conv2d(24, 8, 1, 1, 0)
        )
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.up = nn.Upsample(size=img_size, mode='bilinear')
    
    def forward(self, x_trace):
        x = self.conv(self.pool(self.dilated_conv1(x_trace[1])))
        x = self.pool(torch.cat([x, self.dilated_conv2(x_trace[2])], dim=1))
        x = self.dws_conv(torch.cat([x, self.dilated_conv3(x_trace[3])], dim=1))
        x = self.up(x)
        return x
        
          
class DecoderBlock(nn.Module):
    
    def __init__(self, seq, num_classes=2):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        if seq == 3:
            self.conv = nn.ModuleList([
                ConvBnRelu(8, 8, 3, 1, 1),
                ConvBnRelu(16, num_classes, 3, 1, 1),
            ])
        elif seq == 2:
            self.conv = nn.Sequential(
                ConvBnRelu(16, 16, 3, 1, 1),
                ConvBnRelu(16, 8, 3, 1, 1),
            )
        elif seq == 1:
            self.conv = nn.Sequential(
                DepthSepConv(32, 32, 3, 1, 1),
                DepthSepConv(32, 32, 3, 1, 1),
                ConvBnRelu(32, 16, 3, 1, 1),
            )
        elif seq == 0:
            self.conv = nn.Sequential(
                DepthSepConv(128, 128, 3, 1, 1),
                DepthSepConv(128, 64, 3, 1, 1),
                DepthSepConv(64, 32, 3, 1, 1),
            )
        else:
            raise ValueError('Invalid number of Decoder blocks: expected 4 got %d'%seq)
        
    def forward(self, x, indices, dc_out=None):
        x = self.unpool(x, indices)
        x = self.conv(x) if dc_out is None else \
            self.conv[1](torch.cat([self.conv[0](x), dc_out], dim=1))
        return x
    
    
class DCNet(nn.Module):
    
    def __init__(self, img_size, in_channels, num_classes):
        super().__init__()
        self.encoder = nn.ModuleList([
            EncoderBlock(in_channels, 8, 'full_conv'),
            EncoderBlock(8, 16, 'full_conv'),
            EncoderBlock(16, 32, 'mixed_conv'),
            EncoderBlock(32, 64, 'dws_conv'),
        ])
        self.decoder = nn.ModuleList(
            [DecoderBlock(i, num_classes) for i in range(4)]
        )
        self.dc_block = DCBlock(img_size)
        
    def forward(self, x):
        indices = []
        x_trace = []
        for block in self.encoder:
            x, idx = block(x)
            x_trace.append(x)
            indices.append(idx)
        dc_out = self.dc_block(x_trace)
        for block in self.decoder:
            x = block(x, indices.pop()) if len(indices)>1 else \
                block(x, indices.pop(), dc_out)
        return x
                
        
@arch_entrypoints.register('dcnet')
def build_model(config):
    return DCNet(
        img_size=config.DATA.IMAGE_SIZE,
        in_channels=config.DATA.IN_CHANS,
        num_classes=config.DATA.NUM_CLASSES,
    )
