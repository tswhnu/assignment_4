# Define a baseline from literature
import torch.nn.functional as F
import torch

# # here I chose the U-Net architecture, the first setp is implement the double conv layer of the U-Net
from torch import nn


class Double_conv(nn.Module):
    # (conv->BN->Relu)*2
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_sample(nn.Module):
    # maxpool->double_conv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up_sample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # up sampling function
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # up sampling still need conv layers
        self.conv = Double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 from the up sampling stage and x2 from down sampling stage which has bigger size
        x1 = self.up(x1)

        # using padding let the x1 has the same shape x2
        # first calculate the size of padding
        # the difference in y direction(dim[2])
        diff_y = x2.size()[2] - x1.size()[2]
        # the difference in x direction(dim[3])
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_x - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels, n_class):
        super().__init__()

        self.in_channels = in_channels
        self.n_class = n_class

        self.in_conv = Double_conv(in_channels, 64)
        self.down1 = Down_sample(64, 128)
        self.down2 = Down_sample(128, 256)
        self.down3 = Down_sample(256, 512)
        self.down4 = Down_sample(512, 1024)
        self.up1 = Up_sample(1024, 512)
        self.up2 = Up_sample(512, 256)
        self.up3 = Up_sample(256, 128)
        self.up4 = Up_sample(128, 64)
        self.out_conv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        out = self.out_conv(x9)

        return out
