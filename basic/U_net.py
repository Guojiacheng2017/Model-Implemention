import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Contract(nn.Module):
    def __init__(self, in_channels, out_channels, pad=False):
        super(Contract, self).__init__()
        self.pad = pad
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

    def forward(self, x, last=False):
        height = x.shape[2]
        width = x.shape[3]
        out = self.conv1(x)
        if self.pad is False:
            out = centerCrop(out, height, width)
            # pass
            # output should be cropped here
        rst = out
        if last is False:
            rst = self.pooling(rst)
        return rst, out


class Expansion(nn.Module):
    def __init__(self, in_channels, out_channels, pad=False):
        super(Expansion, self).__init__()
        self.pad = pad
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
        # print(x.shape)
        out = self.deconv1(x)
        # if self.pad is False:
        #     pass
        # rst = np.concatenate((out_c, out), axis=1)
        # print(out_c.shape, out.shape)

        rst = torch.cat((out_c, out), dim=1)
        rst = self.conv2(rst)
        return rst


class Unet(nn.Module):
    def __init__(self, img_channel, pad):
        super(Unet, self).__init__()
        # self.img_size = img_size
        self.imgchannel = img_channel
        self.pad = pad
        # print(self.imgshape)

        self.layer1 = Contract(self.imgchannel, 64, pad=self.pad)
        self.layer2 = Contract(64, 128, pad=self.pad)
        self.layer3 = Contract(128, 256, pad=self.pad)
        self.layer4 = Contract(256, 512, pad=self.pad)
        self.layer = Contract(512, 1024, pad=self.pad)

        self.delayer4 = Expansion(1024, 512, pad=self.pad)
        self.delayer3 = Expansion(512, 256, pad=self.pad)
        self.delayer2 = Expansion(256, 128, pad=self.pad)
        self.delayer1 = Expansion(128, 64, pad=self.pad)

        self.conv = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, img):
        out, l1 = self.layer1(img)
        out, l2 = self.layer2(out)
        out, l3 = self.layer3(out)
        out, l4 = self.layer4(out)
        out, _ = self.layer(out, last=True)

        out = self.delayer4(l4, out)
        out = self.delayer3(l3, out)
        out = self.delayer2(l2, out)
        out = self.delayer1(l1, out)

        rst = self.conv(out)
        return rst

def centerCrop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)    # means {(x.size()[2] or xheight) - height} / 2
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)
    # print(crop_w, crop_h)
    rst = F.pad(x, (crop_w.ceil().int()[0], crop_w.floor().int()[0], crop_h.ceil().int()[0], crop_h.floor().int()[0]))
    return rst



