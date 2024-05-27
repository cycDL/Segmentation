import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table

"""
轻量化注意力模块
"""
class lightWeight_attention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(lightWeight_attention, self).__init__()
        self.pad = kernel_size // 2
        self.conv_1d = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.up(x)
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.conv_1d(x_h)
        x_h = self.gn(x_h)
        x_h = self.sigmoid(x_h).view(b, c, h, 1)

        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.conv_1d(x_w)
        x_w = self.gn(x_w)
        x_w = self.sigmoid(x_w).view(b, c, 1, w)
        return x * x_h * x_w


"""
条形卷积 残差主干网络 
包含s=2下采样以及s=1普通的
依旧是四个阶段
"""
class GCN(nn.Module):
    def __init__(self, in_chan, out_chan, strides=1):
        super(GCN, self).__init__()
        self.downsample = None
        self.stride = strides
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan//2)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 没怎么考虑padding，后期考虑
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1)
        self.conv3x3 = nn.Conv2d(out_chan, out_chan//2, kernel_size=(3, 3), stride=self.stride, padding=1)  # 保证分辨率一致
        self.conv1x5 = nn.Conv2d(out_chan, out_chan//2, kernel_size=(1, 5), padding=(0, 5//2))
        self.conv5x1 = nn.Conv2d(out_chan//2, out_chan//2, kernel_size=(5, 1), padding=(5//2, 0))
        self.conv2 = nn.Conv2d(out_chan//2, out_chan, kernel_size=1, stride=1)
        if in_chan != out_chan or self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    kernel_size=1,  # 侧边使用1*1进行下采样升维
                    stride=self.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.conv3x3(out)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out2 = self.conv1x5(out)
        out2 = self.conv5x1(out2)
        if self.stride != 1:
            out2 = self.max_pool(out2)

        out = out1 + out2
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = out + shortcut
        out_ = self.relu(out_)
        return out_


"""
构建注意力和非对称卷积的 主干网络模块
"""
class lwb_gcn(nn.Module):
    def __init__(self, in_chan, out_chan, strides=1):
        super(lwb_gcn, self).__init__()
        self.stride = strides
        self.lw_attention = lightWeight_attention(in_chan)
        self.GCN = GCN(in_chan, out_chan, strides=self.stride)

    def forward(self, x):
        x = self.lw_attention(x)
        x = self.GCN(x)
        return x


"""
主干网络 输出fpn_list
"""
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.chan_list = [64, 128, 256, 512]  # 这边改通道数
        # 上来就下采样 c2其实可以不先下采样
        self.c2 = nn.Sequential(lwb_gcn(in_chan=32, out_chan=self.chan_list[0], strides=2),
                                lwb_gcn(in_chan=self.chan_list[0], out_chan=self.chan_list[0], strides=1)
                                )
        self.c3 = nn.Sequential(lwb_gcn(in_chan=self.chan_list[0], out_chan=self.chan_list[1], strides=2),
                                lwb_gcn(in_chan=self.chan_list[1], out_chan=self.chan_list[1], strides=1)
                                )
        self.c4 = nn.Sequential(lwb_gcn(in_chan=self.chan_list[1], out_chan=self.chan_list[2], strides=2),
                                lwb_gcn(in_chan=self.chan_list[2], out_chan=self.chan_list[2], strides=1)
                                )
        self.c5 = nn.Sequential(lwb_gcn(in_chan=self.chan_list[2], out_chan=self.chan_list[3], strides=2),
                                lwb_gcn(in_chan=self.chan_list[3], out_chan=self.chan_list[3], strides=1)
                                )

    def forward(self, x):
        fpn_list = []

        x = self.c2(x)
        fpn_list.append(x)

        x = self.c3(x)
        fpn_list.append(x)

        x = self.c4(x)
        fpn_list.append(x)

        x = self.c5(x)
        fpn_list.append(x)
        return fpn_list  # [c2,c3,c4,c5]


"""
特征金字塔FPN构建
主干网络最后输出得fpn_list，有四个特征图，print维度一下
由于设置的是双金字塔，所以不需要三次上采样，C5-C4, C3-C2即可
"""
class FPN(nn.Module):
    def __init__(self, in_chan_list, out_chan):
        super(FPN, self).__init__()
        self.inner_layer = []  # 存放四个阶段的1*1卷积layer
        self.out_layer = []  # 存放四个阶段的3*3卷积layer
        for in_chan in in_chan_list:
            self.inner_layer.append(nn.Conv2d(in_chan, out_chan, kernel_size=1))
            self.out_layer.append(nn.Conv2d(out_chan, out_chan, kernel_size=3))

    def forward(self, x):
        # 此处 x = [c2, c3, c4, c5] 本质上是特征图list
        head_output = []  # 存放p2-p5的四个特征图

        #  C5->c4
        corent_inner = self.inner_layer[-1](x[-1])  # 过1x1卷积，对C5统一通道数操作
        head_output.append(self.out_layer[-1](corent_inner))  # 过3x3卷积得到p5，对统一通道后过的特征进一步融合，加入head_output列表
        # print(self.out_layer[-1](corent_inner).shape)  # 看看p5维度

        pre_inner = corent_inner  # C5经过1*1卷积之后
        corent_inner = self.inner_layer[-2](x[-2])  # c4 经过1*1卷积
        size = corent_inner.shape[2:]  # 获取上采样大小size
        pre_top_down = F.interpolate(pre_inner, size=size)  # C5上采样
        add_pre2cornet = pre_top_down + corent_inner  # add通道数不变
        head_output.append(self.out_layer[-2](add_pre2cornet))  # 生成P4

        # C3->C2 分别 P2的过程
        # 生成P3
        cornet_inner_c3 = self.inner_layer[-3](x[-3])  # 1*1卷积
        pre_inner_c3 = self.out_layer[-3](cornet_inner_c3)  # 3*3卷积
        head_output.append(pre_inner_c3)  # 生成p3

        # 生成P2
        cornet_inner_c2 = self.inner_layer[-4](x[-4])
        size_c2 = cornet_inner_c2.shape[2:]
        pre_top_down_c2 = F.interpolate(cornet_inner_c3, size=size_c2)
        add_pre2cornet_c2 = pre_top_down_c2 + cornet_inner_c2
        head_output.append(self.out_layer[-4](add_pre2cornet_c2))  # 生成P4

        # 已经完成，可以只取P4、P2两个特征图
        # 对P4进行上采样与P2融合
        return head_output  # [p2, p3, p4, p5]


"""
总网络
"""
class LW_DFPN(nn.Module):
    def __init__(self, num_class):
        super(LW_DFPN, self).__init__()
        # stem layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # backbone
        self.backbone = Backbone()

        # 双FPN
        self.in_chan_list = [64, 128, 256, 512]  # 此处更改
        self.out_chan_fpn = 128  # 分割模块也要对应改
        self.fpn = FPN(in_chan_list=self.in_chan_list, out_chan=self.out_chan_fpn)

        # 对P4进行上采样与P2融合

        # segmentation 模块
        self.conv3_3 = nn.Conv2d(in_channels=self.out_chan_fpn*2, out_channels=self.out_chan_fpn, kernel_size=(3, 3),
                                 stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_chan_fpn)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv1_1 = nn.Conv2d(in_channels=self.out_chan_fpn, out_channels=num_class, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.backbone(x)

        x = self.fpn(x)  # [p2, p3, p4, p5]

        x_high = F.interpolate(x[-2], scale_factor=4, mode='bilinear', align_corners=True)
        x_low = x[-4]  # 128*64*64
        x = torch.cat((x_high, x_low), dim=1)  # 256*64*64
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)  # 256*256*256

        x = self.conv3_3(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv1_1(x)
        return x


"""
1.构建注意力和非对称卷积的 主干网络 输出是fpn_list  # 已经OK，主干网络模块+主干网络 完成
2.FPN构建 注意是分开的不能 三个上采样改为2个  # 已经OK，可以只取P4、P2两个特征图
3.写总网络，分开的四个特征后续，还要上采样融合解码，具体要看PFANet  # 对P4进行上采样与P2融合

注：paddle例子是分开输入输出，这边写成总的封装一下
"""

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    # # 创建 输入网络的tensor
    # tensor = (x, )

    model = LW_DFPN(3)

    # # 分析FLOPs
    # flops = FlopCountAnalysis(model, tensor)
    # print("G_FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print("模型参数量", parameter_count_table(model))

    y = model(x)
    print(y.shape)