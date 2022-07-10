import math
import torch
import torch.nn.functional as F
from torch import nn
from utils import CE_Loss
from torchvision import models
from resnet import resnet50

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super().__init__()
        reduction_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, reduction_channels, pool_size) for pool_size in pool_sizes])

    def _make_stages(self, in_channels, out_channels, bin_sz):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=bin_sz),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        
        pyramids = [features]
        for stage in self.stages:
            pyramids.append(
                F.interpolate(stage(features), size=features.shape[2:], mode='bilinear', align_corners=True)
            )
        output = torch.cat(pyramids, dim=1)
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()

        resnet = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu, 
            resnet.conv2,
            resnet.bn2, 
            resnet.relu, 
            resnet.conv3, 
            resnet.bn3, 
            resnet.relu, 
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer3.named_modules():    
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:             # 取消下采样, PSPNet默认使用的Resnet的下采样率是8
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:             # 取消下采样, PSPNet默认使用的Resnet的下采样率是8
                m.stride = (1, 1)

        out_channel = 2048

        self.pmp = PSPModule(out_channel, pool_sizes=[1, 2, 3, 6])
        
        self.cls = nn.Sequential(
            nn.Conv2d(out_channel * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, padding=1, bias=False)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        output = self.pmp(x)
        output = self.cls(output)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

        return output
