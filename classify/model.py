from torch import nn
import torch.nn.functional as F
from torchvision.models import detection

class WhatBirdie(nn.Module):
    def __init__(self):
        super().__init__()

        #input: 3 x 224 x 224
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.res1 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        
        self.conv2 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.res2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.res3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.FConnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 700),
            nn.ReLU(),
            nn.Linear(700, 700),
            nn.ReLU(),
            nn.Linear(700, 325)
        )
        
        
    def forward(self, xb):
        #input: 3 x 224 x 224
        out = F.relu(self.bn1(self.conv1(xb)))   # output: 6 x 224 x224
        out = self.pool2(out)   # output: 6 x 112 x 112
        out = self.res1(out) + out   # output: 6 x 112 x 112
       
        out = F.relu(self.bn2(self.conv2(out)))   # output: 32 x 112 x 112
        out = self.pool2(out)    # output: 32 x 56 x 56
        out = self.res2(out) + out   # output: 32 x 56 x 56
       
        out = F.relu(self.bn3(self.conv3(out)))   # output: 64 x 56 x 56
        out = self.pool4(out)    # output: 64 x 14 x 14
        out = self.res3(out) + out    # output: 64 x 14 x 14
         
        out = F.relu(self.bn4(self.conv4(out)))   # output: 128 x 14 x 14
        out = self.pool2(out)    # output: 128 x 7 x 7

        out = self.FConnected(out)   

        return out

frcnn = detection.fasterrcnn_resnet50_fpn(pretrained=True)