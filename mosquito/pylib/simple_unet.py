import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, features: int = 64):
        super().__init__()

        self.input = self.double_conv(in_channels, features)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = self.block(features, features * 2)
        self.encoder2 = self.block(features * 2, features * 4)
        self.encoder3 = self.block(features * 4, features * 8)
        self.encoder4 = self.block(features * 8, features * 16)

        self.bottleneck = nn.Conv2d(features * 16, features * 16, kernel_size=3)

        self.unpool4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self.block((features * 8) * 2, features * 8)

        self.unpool3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self.block((features * 4) * 2, features * 4)

        self.unpool2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self.block((features * 2) * 2, features * 2)

        self.unpool1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self.block(features * 2, features)

        self.output = nn.Conv2d(features, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.input(x)
        enc1 = self.pool(self.encoder1(x))
        enc2 = self.pool(self.encoder2(enc1))
        enc3 = self.pool(self.encoder3(enc2))
        enc4 = self.pool(self.encoder4(enc3))

        x = self.bottleneck(enc4)

        x = self.unpool4(x)
        x = self.decoder4(torch.cat([x, enc4], dim=1))

        x = self.unpool3(x)
        x = self.decoder3(torch.cat([x, enc3], dim=1))

        x = self.unpool2(x)
        x = self.decoder2(torch.cat([x, enc2], dim=1))

        x = self.unpool1(x)
        x = self.decoder1(torch.cat([x, enc1], dim=1))

        x = self.output(x)
        return x

    @staticmethod
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
