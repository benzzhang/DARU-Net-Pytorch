import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UNet']


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class conv_final(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_final, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, bilinear=True):
        super(UNet, self).__init__()

        self.enconv1 = DoubleConv(1, 32)
        self.enconv2 = DoubleConv(32, 64)
        self.enconv3 = DoubleConv(64, 128)
        self.enconv4 = DoubleConv(128, 256)
        self.enconv5 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)

        self.deconv4 = DoubleConv(512, 256)
        self.deconv3 = DoubleConv(256, 128)
        self.deconv2 = DoubleConv(128, 64)
        self.deconv1 = DoubleConv(64, 32)

        self.out = conv_final(32, 1)

    def forward(self, x):
        conv1 = self.enconv1(x)
        pool1 = self.maxpool(conv1)

        conv2 = self.enconv2(pool1)
        pool2 = self.maxpool(conv2)

        conv3 = self.enconv3(pool2)
        pool3 = self.maxpool(conv3)

        conv4 = self.enconv4(pool3)
        pool4 = self.maxpool(conv4)

        conv5 = self.enconv5(pool4)  # (512, 32, 32)

        up4 = torch.cat([self.up1(conv5), conv4], dim=1)
        deconv4 = self.deconv4(up4)
        up3 = torch.cat([self.up2(deconv4), conv3], dim=1)
        deconv3 = self.deconv3(up3)
        up2 = torch.cat([self.up3(deconv3), conv2], dim=1)
        deconv2 = self.deconv2(up2)
        up1 = torch.cat([self.up4(deconv2), conv1], dim=1)
        deconv1 = self.deconv1(up1)

        out = self.out(deconv1)

        return out
