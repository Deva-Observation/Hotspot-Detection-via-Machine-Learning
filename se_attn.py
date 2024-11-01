import numpy as np
import torch
from torch import nn
from torch.nn import init

class SE_module(nn.Module):

    def __init__(self, channel, scaling):
        super(SE_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#把输入池化为1*1*C的尺寸
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // scaling, bias=False),#C->C/r
            nn.ReLU(inplace = True),#激活函数使用ReLU
            nn.Linear(channel // scaling, channel, bias=False),#C/r->C
            nn.Sigmoid()#激活函数使用Sigmoid
        )

    def forward(self, x):
        b, c, _, _=x.size()#获取batch大小和通道数
        y = self.avg_pool(x).view(b, c)#对输入x进行全局池化
        y = self.fc(y).view(b, c, 1, 1)#进行全连接
        return x * y.expand_as(x)#sigmoid激活后的参数作为权重与原输入相乘，获得注意力机制后的特征图
    

if __name__ == "__main__":
    input = torch.randn(64, 256, 8, 8)
    model = SE_module(channel=256, scaling=8)
    output = model(input)
    print(output.shape)
        