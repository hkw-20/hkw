import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 确保in_planes大于ratio
        if in_planes < ratio:
            ratio = max(1, in_planes // 2)  # 防止除数为0
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False)  # 确保至少1通道
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 动态获取输入通道数
        in_planes = x.size(1)
        
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 打印通道信息以便调试
        print(f"初始化CBAM模块，输入通道数: {channels}")
        
        # 确保通道数有效
        if channels <= 0:
            raise ValueError(f"无效的通道数: {channels}")
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 打印实际输入维度
        print(f"CBAM输入维度: {x.shape}")
        
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out