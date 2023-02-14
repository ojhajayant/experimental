#!/usr/bin/env python
"""
EVA8_session8_assignment_model.py: CNN class definition for EVA8 assignment 6
"""
from __future__ import print_function
import sys
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./')


class HighwayBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HighwayBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                               padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)

        return x


class Layer(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 skip_resblock: bool = False):
        super(Layer, self).__init__()
        self.highway_block = HighwayBlock(input_size, output_size)
        self.skip_resblock = skip_resblock
        if not skip_resblock:
            self.res_block = ResBlock(output_size, output_size)

    def forward(self, x):
        out = self.highway_block(x)
        if not self.skip_resblock:
            out += self.res_block(out)
        return out


class EVA8_session8_assignment_model(nn.Module):
    def __init__(self, num_classes):
        super(EVA8_session8_assignment_model, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = Layer(64, 128)
        self.layer2 = Layer(128, 256, skip_resblock=True)
        self.layer3 = Layer(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc = nn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
