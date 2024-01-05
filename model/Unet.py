import torch
import torch.nn as nn
import torch.nn.functional as F

def double_con3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = double_con3x3(in_channels=in_channels, out_channels=out_channels)
    def forward(self, input):
        x = self.maxpool(input)
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.conv = double_con3x3(in_channels=in_channels, out_channels=out_channels)
    def forward(self, input1, input2):
        input1 = F.interpolate(input1, size=input2.size()[2:], mode='bilinear', align_corners=True)
        output = torch.cat((input1, input2), dim=1)  # 保证拼接
        output = self.conv(output)
        return output


class Unet(nn.Module):
    def __init__(self, num_class):
        super(Unet, self).__init__()
        self.inconv = double_con3x3(in_channels=3, out_channels=64)
        self.down1 = down(in_channels=64, out_channels=128)
        self.down2 = down(in_channels=128, out_channels=256)
        self.down3 = down(in_channels=256, out_channels=512)
        self.down4 = down(in_channels=512, out_channels=512)
        self.up1 = up(in_channels=1024, out_channels=256)
        self.up2 = up(in_channels=512, out_channels=128)
        self.up3 = up(in_channels=256, out_channels=64)
        self.up4 = up(in_channels=128, out_channels=64)
        self.outconv = nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, input):
        x1 = self.inconv(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up1(x5, x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        return x

if __name__ =='__main__':
    x = torch.randn(1, 3, 224, 224)
    model = Unet(3)
    y = model(x)
    print(y.shape)




