import torch
import torch.nn as nn

__all__ = ['DARU_Net']


class PPM_Block(nn.Module):
    def __init__(self, in_channels):
        super(PPM_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=8)
        self.upsample3 = nn.Upsample(scale_factor=16)
        self.upsample4 = nn.Upsample(scale_factor=32)
        self.act = nn.ReLU(inplace=True)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        conv1 = self.act(self.conv(self.pool(x, x.shape[-2] // 4)))
        conv2 = self.act(self.conv(self.pool(x, x.shape[-2] // 8)))
        conv3 = self.act(self.conv(self.pool(x, x.shape[-2] // 16)))
        conv4 = self.act(self.conv(self.pool(x, x.shape[-2] // 32)))

        up1 = self.upsample1(conv1)
        up2 = self.upsample2(conv2)
        up3 = self.upsample3(conv3)
        up4 = self.upsample4(conv4)

        return torch.cat((up1, up2, up3, up4), dim=1)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.conv_only = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)

        self.SE = SE_Block(ch_out)
        self.BN = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_only(x1)
        x = self.SE(x2)
        res_sum = torch.add(x, x1)

        return self.act(self.BN(res_sum))


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DARU_Net(nn.Module):
    def __init__(self, in_channels=1):
        super(DARU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.in1 = conv_block(ch_in=in_channels, ch_out=32)
        self.Conv1 = conv_block(ch_in=32, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_Conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_Conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_Conv2 = conv_block(ch_in=128, ch_out=64)

        self.Up1 = up_conv(ch_in=64, ch_out=32)
        self.Up_Conv1 = conv_block(ch_in=64, ch_out=32)

        self.out1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.ppm = PPM_Block(512)

    def forward(self, x):
        # encoding path
        x1 = self.in1(x)       # x1 - (512,512,32)
        mp = self.Maxpool(x1)
        ap = self.Avgpool(x1)
        p1 = torch.add(mp, ap) # p1 - (256,256,32)

        x2 = self.Conv1(p1)    # x2 - (256,256,64)
        mp = self.Maxpool(x2)
        ap = self.Avgpool(x2)
        p2 = torch.add(mp, ap) # p2 - (128,128,64)

        x3 = self.Conv2(p2)    # x3 - (128,128,128)
        mp = self.Maxpool(x3)
        ap = self.Avgpool(x3)
        p3 = torch.add(mp, ap) # p3 - (64,64,128)

        x4 = self.Conv3(p3)    # x4 - (64,64,256)
        mp = self.Maxpool(x4)
        ap = self.Avgpool(x4)
        p4 = torch.add(mp, ap) # p4 - (32,32,256)

        x5 = self.Conv4(p4)
        x5 = self.ppm(x5)      # x5 - (32,32,512)

        # decoding & concat path
        d4 = self.Up4(x5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_Conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_Conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_Conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_Conv1(d1)

        d = self.out1(d1)

        return torch.sigmoid(d)
