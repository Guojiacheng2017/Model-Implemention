import torch
import numpy as np
import torch.nn as nn

class Contract(nn.Module):
    def __init__(self, in_channels, out_channels, pad=False):
        super(Contract, self).__init__()
        self.padding = 1 if pad is True else 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.conv1(x)
        if self.pad is False:
            pass
            # output should be cropped here
        rst = self.pooling(out)
        return rst, out


class Expansion(nn.Module):
    def __init__(self, in_channels, out_channels, pad=False):
        super(Expansion, self).__init__()
        self.padding = 1 if pad is True else 0
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, out_c, x):
        out = self.deconv1(x)
        # if self.pad is False:
        #     pass
        rst = np.concatenate(out_c, out)
        # rst = torch.cat((out_c, out), dim=1)
        rst = self.conv2(rst)
        return rst


class Unet(nn.Module):
    def __init__(self, img):
        super(Unet, self).__init__()
        # self.img_size = img_size
        self.imgshape = img.shape()
        print(self.imgshape)

        self.layer1 = Contract(1, 64, pad=True)
        self.layer2 = Contract(64, 128, pad=True)
        self.layer3 = Contract(128, 256, pad=True)
        self.layer4 = Contract(256, 512, pad=True)
        self.layer = Contract(512, 1024, pad=True)

        self.delayer4 = Expansion(1024, 512, pad=True)
        self.delayer3 = Expansion(512, 256, pad=True)
        self.delayer2 = Expansion(256, 128, pad=True)
        self.delayer1 = Expansion(128, 64, pad=True)

        self.conv = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, img):
        out, l1 = self.layer1(img)
        out, l2 = self.layer2(out)
        out, l3 = self.layer3(out)
        out, l4 = self.layer4(out)
        out, _ = self.layer(out)

        out = self.delayer4(l4, out)
        out = self.delayer3(l3, out)
        out = self.delayer2(l2, out)
        out = self.delayer1(l1, out)

        rst = self.conv(out)
        return rst





