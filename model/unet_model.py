# full assembly of the sub-parts to form the complete net
import torch
import torch.nn.functional as F
import torch.nn as nn

# from .unet_parts import *
# import sys
# sys.path.append('.')
from model.unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
#         return F.sigmoid(x)
        return x

# if __name__ == '__main__':
#     # from thop import profile
#     import warnings
#     warnings.filterwarnings('ignore')
#
#     model = UNet( n_classes=32).cuda()
#     x1 = torch.rand(1, 3, 512, 512).cuda()
#     x2 = torch.rand(1, 3, 512, 512).cuda()
#     out = model(x1)
#     print('out:',out.size())
#     flops, params = profile(model, input_size=(1, 3, 512, 512))
#     print("FLOPs :{:.3f}G\nparams:{:.3f}M".format(flops / 1e9, params / 1e6))


