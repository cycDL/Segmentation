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
        output1 = torch.cat((input1,input2),dim=1)
        output2 = self.conv(output1)
        return output2,output1

class BTSmodule(nn.Module):
    def __init__(self, in_channels, out_channels, outputsize=224):
        super(BTSmodule, self).__init__()
        self.outputsize = outputsize
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, input):
        x = self.conv1x1(input)
        x = F.interpolate(x, size=self.outputsize, mode='bilinear', align_corners=True)
        return x


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
        self.outconv = nn.Conv2d(in_channels=64,out_channels=num_class,kernel_size=1,stride=1,padding=0,bias=False)
        self.BTS2 = BTSmodule(in_channels=1024, out_channels=num_class)
        self.BTS3 = BTSmodule(in_channels=512, out_channels=num_class)
        self.BTS4 = BTSmodule(in_channels=256, out_channels=num_class)
        self.BTS5 = BTSmodule(in_channels=128, out_channels=num_class)
    def forward(self, input):
        x1 = self.inconv(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output1, output2 = self.up1(x5, x4)
        output1, output3 = self.up2(output1,x3)
        output1, output4 = self.up3(output1,x2)
        output1, output5 = self.up4(output1, x1)
        output1 = self.outconv(output1)
        output2 = self.BTS2(output2)
        output3 = self.BTS3(output3)
        output4 = self.BTS4(output4)
        output5 = self.BTS5(output5)
        return output1, output2, output3, output4, output5

if __name__ =='__main__':
    x = torch.randn(1,3,224,224)
    model = Unet(3)
    y = model(x)
    print(y[2].shape)




