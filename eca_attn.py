import numpy as np
import torch
from torch import nn
from torch.nn import init

class ECA_module(nn.Module):
    
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_module, self).__init__()
        kernel_size = int(abs((np.log2(channel) + b) / gamma))
        #确定卷积核的尺寸
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
         #确保卷积核尺寸始终是奇数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #将输入池化为1*1*C的形式
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        #进行一维卷积，输入输出通道数为1，设置same填充令输入输出大小一致
        self.sigmoid = nn.Sigmoid()
        #使用sigmoid函数作为激活函数

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        '''
        y.squeeze(-1)：将 y 在最后一个维度上去掉大小为 1 的维度。假设 y 的形状为 (batch_size, channels, length, 1)，则 squeeze(-1) 会将其变为 (batch_size, channels, length)。
        .transpose(-1, -2)：将最后两个维度对换。现在 y 的形状从 (batch_size, channels, length) 变成 (batch_size, length, channels)。这样做是为了适应 Conv1d 层的输入格式，通常要求 [batch_size, in_channels, length]。
        self.conv(...)：对 y 进行一维卷积操作。输出的形状会是 (batch_size, 1, length)。
        .transpose(-1, -2)：再次将最后两个维度对换，使其变成 (batch_size, length, 1)，方便后续操作。
        .unsqueeze(-1)：在最后一个维度增加一个大小为 1 的维度，使 y 的最终形状变为 (batch_size, length, 1, 1)，以恢复为四维张量。
        '''
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
if __name__ == "__main__":
    input = torch.randn(64, 256, 8, 8)
    model = ECA_module(channel=256)
    output = model(input)
    print(output.shape)