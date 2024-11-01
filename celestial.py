import numpy as np
import torch
from torch import nn
from torch.nn import init
import eca_attn
import se_attn

class CelestialNet(nn.Module):

    def __init__(self):
        super(CelestialNet, self).__init__()
        self.model = nn.Sequential(
            #初始尺寸[224,224,3]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            #[224,224,64]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #[112,112,64]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            #[112,112,64]
            se_attn.SE_module(channel=64, scaling=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #[56,56,64]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, bias=False),
            #[56,56,128]
            se_attn.SE_module(channel=128, scaling=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #[28,28,128]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, bias=False),
            #[28,28,256]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            #[28,28,256]
            se_attn.SE_module(channel=256, scaling=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #[14,14,256]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, bias=False),
            #[14,14,512]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=False),
            #[14,14,512]
            se_attn.SE_module(channel=512, scaling=2),
            eca_attn.ECA_module(channel=512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            #[1,1,512]
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),

        )

    def forward(self, x):
        x = self.model(x)
        return x