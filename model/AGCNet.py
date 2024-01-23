import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F
from model.ppliteseg_stdc import STDCNet813
from model.ppliteseg_UAFM import UAFM_SpAtten, ConvBNReLU


class DACM(nn.Module):
    """
    Daliated Asymmetric Convolution Module: DACM 空洞非对称卷积模块
    """
    def __int__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(DACM, self).__int__()

        # # 先拆分通道
        # # split按照具体数量拆分 chunk按照块数来
        # self.chunk1 = torch.chunk(chunks=2, dim=2)  # 按通道维度分为2块 x1,x2
        # # 拼接通道
        # self.cat1 = torch.cat(dim=2)  # 按通道拼接

        # 非对称卷积 先k*1 再1*k
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size(3, 1),
                                 stride=stride, padding=(padding, 0), dilation=dilation,
                                 groups=groups, bias=bias)  # padding看横纵轴, 空洞率是核心
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size(1, 3),
                                 stride=stride, padding=(0, padding), dilation=dilation,
                                 groups=groups, bias=bias)

        self.conv3_3 = nn.Sequential(self.conv3_1, self.conv1_3)  # 可能缺乏BN RELU层

        self.conv5_1 = nn.Conv2d(in_channels, out_channels, kernel_size(5, 1),
                                 stride=stride, padding=(padding, 0), dilation=dilation,
                                 groups=groups, bias=bias)
        self.conv1_5 = nn.Conv2d(in_channels, out_channels, kernel_size(1, 5),
                                 stride=stride, padding=(0, padding), dilation=dilation,
                                 groups=groups, bias=bias)
        self.conv5_5 = nn.Sequential(self.conv5_1, self.conv1_5)

    def forward(self, x):
        input1 = x
        x1, x2 = torch.chunk(x, chunks=2, dim=2)  # 拆分2块
        x1 = self.conv3_3(x1)
        x2 = self.conv5_5(x2)
        x = torch.cat((x1, x2), dim=2)  # 拼接
        return input1 + x  # 残差相加



class PPContextModule(nn.Module):
    """
    Simple Context module. SPPM 简单金字塔模块
    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = ConvBNReLU(
            in_planes=inter_channels,
            out_planes=out_channels,
            kernel=3)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNReLU(
            in_planes=in_channels, out_planes=out_channels, kernel=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out

class ResPathModule(nn.Module):
    def __int__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(ResPathModule, self).__int__()

        # 非对称卷积 先k*1 再1*k
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size(3, 1),
                                 stride=stride, padding=(padding, 0), dilation=dilation,
                                 groups=groups, bias=bias)  #
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size(1, 3),
                                 stride=stride, padding=(0, padding), dilation=dilation,
                                 groups=groups, bias=bias)

        self.conv3_3 = nn.Sequential(self.conv3_1, self.conv1_3)  # 可能缺乏BN RELU层

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        x1 = self.conv3_3(x)
        x2 = self.conv1_1(x)
        out = x1 + x2
        return out

class AGCNet(nn.Module):
    """
    整体网络架构
    解码器部分还没写，先用普通的，后面在创新改改
    """
    def __init__(self):
        super(AGCNet, self).__init__()
    def forward(self, x):
        return x