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

# class up(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(up, self).__init__()
#         self.conv = double_con3x3(in_channels=in_channels, out_channels=out_channels)
#     def forward(self, input1, input2):
#         input1 = F.interpolate(input1, size=input2.size()[2:], mode='bilinear', align_corners=True)
#         output = torch.cat((input1,input2),dim=1)
#         output = self.conv(output)
#         return output

class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        input = self.conv1x1(input)
        x = self.conv3x3(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3x3(x)
        output = x+input
        output = self.relu(output)
        return output


class RRB_Unet(nn.Module):
    def __init__(self, num_class):
        super(RRB_Unet, self).__init__()
        self.inconv = double_con3x3(in_channels=3, out_channels=64)
        self.down1 = down(in_channels=64, out_channels=128)
        self.down2 = down(in_channels=128, out_channels=256)
        self.down3 = down(in_channels=256, out_channels=512)
        self.down4 = down(in_channels=512, out_channels=512)
        self.RRB1 = RRB(in_channels=512, out_channels=num_class)
        self.RRB2 = RRB(in_channels=512, out_channels=num_class)
        self.RRB3 = RRB(in_channels=256, out_channels=num_class)
        self.RRB4 = RRB(in_channels=128, out_channels=num_class)
        # self.RRB5 = RRB(in_channels=64, out_channels=num_class)
        self.RRB = RRB(in_channels=num_class, out_channels=num_class)
        # self.up1 = up(in_channels=num_class*2, out_channels=num_class)
        # self.up2 = up(in_channels=num_class*2, out_channels=num_class)
        # self.up3 = up(in_channels=num_class*2, out_channels=num_class)
        # self.up4 = up(in_channels=num_class*2, out_channels=num_class)
        self.outconv = nn.Conv2d(in_channels=num_class,out_channels=num_class,kernel_size=1,stride=1,padding=0,bias=False)
    def forward(self, input):
        x1 = self.inconv(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.RRB1(x5)
        x4 = self.RRB2(x4)
        x3 = self.RRB3(x3)
        x2 = self.RRB4(x2)
        # x1 = self.RRB5(x1)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = x5+x4
        x = self.RRB(x)
        x = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = x+x3
        x = self.RRB(x)
        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = x+x2
        x = self.RRB(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

if __name__ =='__main__':
    x = torch.randn(1,3,224,224)
    model = RRB_Unet(3)
    y = model(x)
    print(y.shape)
