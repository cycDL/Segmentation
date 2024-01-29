import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F
from model.ppliteseg_stdc import STDCNet813
from model.ppliteseg_UAFM import UAFM_SpAtten, ConvBNReLU


class Conv2d_batchnorm(torch.nn.Module):
    '''
    普通卷积模块-->conv+BN+relu （标准模块）
    其实没用到
    Arguments:
        num_in_filters {int} -- number of input filters
        num_out_filters {int} -- number of output filters
        kernel_size {tuple} -- size of the convolving kernel
        stride {tuple} -- stride of the convolution (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})

    '''

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                                     stride=stride, padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        else:
            return x

class DACM(nn.Module):
    """
    Daliated Asymmetric Convolution Module: DACM 空洞非对称卷积模块
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(DACM, self).__init__()

        # # 先拆分通道
        # # split按照具体数量拆分 chunk按照块数来
        # self.chunk1 = torch.chunk(chunks=2, dim=2)  # 按通道维度分为2块 x1,x2
        # # 拼接通道
        # self.cat1 = torch.cat(dim=2)  # 按通道拼接

        self.batchnorm = torch.nn.BatchNorm2d(out_channels)


        # 非对称卷积 先k*1 再1*k
        self.conv3_1 = nn.Conv2d(in_channels//2, out_channels//2, kernel_size(3, 1),
                                 stride=stride, padding=(padding, 0), dilation=dilation,
                                 groups=groups, bias=bias)  # padding看横纵轴, 空洞率是核心
        self.conv1_3 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size(1, 3),
                                 stride=stride, padding=(0, padding), dilation=dilation,
                                 groups=groups, bias=bias)

        self.conv3_3 = nn.Sequential(self.conv3_1, self.conv1_3)  # 可能缺乏BN RELU层


        self.conv5_1 = nn.Conv2d(in_channels//2, out_channels//2, kernel_size(5, 1),
                                 stride=stride, padding=(padding, 0), dilation=dilation,
                                 groups=groups, bias=bias)
        self.conv1_5 = nn.Conv2d(out_channels//2, out_channels//2, kernel_size(1, 5),
                                 stride=stride, padding=(0, padding), dilation=dilation,
                                 groups=groups, bias=bias)
        self.conv5_5 = nn.Sequential(self.conv5_1, self.conv1_5)

    def forward(self, x):
        input1 = x
        x1, x2 = torch.chunk(x, chunks=2, dim=0)  # 拆分2块 dim=0 三维数据按照深度方向

        x1 = self.conv3_3(x1)
        x1 = self.batchnorm(x1)
        x1 = nn.ReLU(x1)

        x2 = self.conv5_5(x2)
        x2 = self.batchnorm(x2)
        x2 = nn.ReLU(x2)
        x = torch.cat((x1, x2), dim=0)  # 拼接
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
    """
    先废弃了
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(ResPathModule, self).__init__()

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


class Respath(torch.nn.Module):
    '''
    ResPath

    Arguments:
        num_in_filters {int} -- Number of filters going in the respath
        num_out_filters {int} -- Number of filters going out the respath
        respath_length {int} -- length of ResPath

    '''

    def __init__(self, in_channels, out_channels, stride=1,
                padding=0, dilation=1, groups=1, bias=True, respath_length=None):

        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        # 决定用4,3,2,1 ResPath模块
        for i in range(self.respath_length):
            if (i == 0):
                self.shortcuts.append(
                    Conv2d_batchnorm(in_channels, out_channels, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1),
                              stride=stride, padding=(padding, 0), dilation=dilation,
                              groups=groups, bias=bias))
                self.convs.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3),
                              stride=stride, padding=(0, padding), dilation=dilation,
                              groups=groups, bias=bias))


            else:
                self.shortcuts.append(
                    Conv2d_batchnorm(out_channels, out_channels, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1),
                              stride=stride, padding=(padding, 0), dilation=dilation,
                              groups=groups, bias=bias))
                self.convs.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3),
                              stride=stride, padding=(0, padding), dilation=dilation,
                              groups=groups, bias=bias))

            self.bns.append(torch.nn.BatchNorm2d(out_channels))

    def forward(self, x):

        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

        return x

class AGCNet(nn.Module):
    """
    整体网络架构
    解码器和multiResNet一样用和编码器的DACM
    """
    def __init__(self, input_channels, num_classes, alpha=1.5):
        super(AGCNet, self).__init__()
        # 通道超参数alpha 原文就是如此设置的
        self.alpha = alpha
        # Encoder Path
        self.DACM1_1 = DACM(input_channels, 32)
        self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha*0.5)  # 32*alpha
        self.pool1 = nn.MaxPool2d(2)
        self.repath1 = Respath(self.in_filters1, 32, respath_length=4)

        self.DACM1_2 = DACM(self.in_filters1, 32*2)
        self.in_filters2 = int(32*2 * self.alpha * 0.167) + int(32*2 * self.alpha * 0.333) + int(32*2 * self.alpha*0.5)
        self.pool2 = nn.MaxPool2d(2)
        self.repath2 = Respath(self.in_filters2, 32*2, respath_length=3)

        self.DACM1_3 = DACM(self.in_filters2, 32 * 4)
        self.in_filters3 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
            32 * 4 * self.alpha * 0.5)
        self.pool3 = nn.MaxPool2d(2)
        self.repath3 = Respath(self.in_filters3, 32 * 4, respath_length=2)

        self.DACM1_4 = DACM(self.in_filters3, 32 * 8)
        self.in_filters4 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
            32 * 8 * self.alpha * 0.5)
        self.pool4 = nn.MaxPool2d(2)
        self.repath4 = Respath(self.in_filters4, 32 * 8, respath_length=1)
        # 此处SPPM 要着重写
        self.in_filters5 = int(32 * 16 * self.alpha * 0.167) + int(32 * 16 * self.alpha * 0.333) + int(
            32 * 16 * self.alpha * 0.5)  # 可能是SPPM的通道数

        # Decoder Path
        self.upsample4 = nn.ConvTranspose2d(self.in_filters5, 32*8, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters4 = 32*8 * 2
        self.DACM2_4 = DACM(self.concat_filters4, 32*8)

        self.upsample3 = nn.ConvTranspose2d(self.in_filters4, 32 * 4, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters3 = 32*4 * 2
        self.DACM2_3 = DACM(self.concat_filters3, 32 * 4)

        self.upsample2 = nn.ConvTranspose2d(self.in_filters3, 32 * 2, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters2 = 32 * 2 * 2
        self.DACM2_2 = DACM(self.concat_filters2, 32 * 2)

        self.upsample1 = nn.ConvTranspose2d(self.in_filters2, 32, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters1 = 32 * 2
        self.DACM2_1 = DACM(self.concat_filters1, 32)

        self.conv_final = Conv2d_batchnorm(32, num_classes + 1, kernel_size=(1, 1), activation='None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_DACM1_1 = self.DACM1_1(x)
        x_pool1 = self.pool1(x_DACM1_1)
        x_multires1_1 = self.repath1(x_DACM1_1)

        x_DACM1_2 = self.DACM1_2(x_pool1)
        x_pool2 = self.pool1(x_DACM1_2)
        x_multires1_2 = self.repath1(x_DACM1_2)

        x_DACM1_3 = self.DACM1_3(x_pool2)
        x_pool3 = self.pool1(x_DACM1_3)
        x_multires1_3 = self.repath1(x_DACM1_3)

        x_DACM1_4 = self.DACM1_4(x_pool3)
        x_pool4 = self.pool1(x_DACM1_4)
        x_multires1_4 = self.repath1(x_DACM1_4)

        x_sppm = PPContextModule(256, 256, 512, 2)(x_pool4)  # 前文还要定义

        up4 = torch.cat([self.upsample4(x_sppm), x_multires1_4], axis=1)
        x_multires2_4 = self.DACM2_4(up4)

        up3 = torch.cat([self.upsample3(x_multires2_4), x_multires1_3], axis=1)
        x_multires2_3 = self.DACM2_3(up3)

        up2 = torch.cat([self.upsample2(x_multires2_3), x_multires1_2], axis=1)
        x_multires2_2 = self.DACM2_2(up2)

        up1 = torch.cat([self.upsample1(x_multires2_2), x_multires1_1], axis=1)
        x_multires2_1 = self.DACM2_1(up1)

        out = self.conv_final(x_multires2_1)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = AGCNet(3, 3)
    y = model(x)
    print(y.shape)