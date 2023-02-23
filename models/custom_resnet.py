# Import the necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the ResBlock class

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixBlock, self).__init__()

        self.convMaxPool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.residual_block = ResBlock(out_channels, out_channels)

    def forward(self, x):
        X = self.convMaxPool(x)
        R = self.residual_block(X)
        out = X + R
        return out

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = MixBlock(64, 128)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = MixBlock(256, 512)

        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

