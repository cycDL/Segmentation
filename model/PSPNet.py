"""
作者：咫尺小厘米
链接：https://zhuanlan.zhihu.com/p/115004020
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。"
"""
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F


class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape
        output = F.upsample_bilinear(self.features(x), size[2:])
        return output


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PSPNet, self).__init__()
        print("initializing model")
        # ResNet50效果 = UNet
        self.resnet = models.resnet50(pretrained=pretrained)

        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)

        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

        initialize_weights(self.layer5a, self.layer5b, self.layer5c, self.layer5d, self.final)

    def forward(self, x):
        size = x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))

        return F.upsample_bilinear(x, size[2:])


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = PSPNet(3)
    y = model(x)
    print(y.shape)
