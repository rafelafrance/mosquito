import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, features=64):
        super().__init__()

        self.out_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode1 = self.block(in_channels, features)
        self.encode2 = self.block(features, features * 2)
        self.encode3 = self.block(features * 2, features * 4)
        self.encode4 = self.block(features * 4, features * 8)

        self.bottleneck = self.block(features * 8, features * 16)

        self.unpool4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode4 = self.block(features * 16, features * 8)

        self.unpool3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode3 = self.block(features * 8, features * 4)

        self.unpool2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode2 = self.block(features * 4, features * 2)

        self.unpool1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode1 = self.block(features * 2, features)

        self.output = nn.ConvTranspose2d(
            features, out_channels, kernel_size=2, stride=2
        )

        self.squash_layer = nn.Sigmoid() if out_channels == 1 else nn.Softmax()

    def forward(self, x):
        enc1 = self.pool(self.encode1(x))
        enc2 = self.pool(self.encode2(enc1))
        enc3 = self.pool(self.encode3(enc2))
        enc4 = self.pool(self.encode4(enc3))

        x = self.bottleneck(self.pool(enc4))

        x = self.unpool4(x)
        x = self.decode4(torch.cat((x, enc4), dim=1))

        x = self.unpool3(x)
        x = self.decode3(torch.cat((x, enc3), dim=1))

        x = self.unpool2(x)
        x = self.decode2(torch.cat((x, enc2), dim=1))

        x = self.unpool1(x)
        x = self.decode1(torch.cat((x, enc1), dim=1))

        x = self.output(x)

        return x

    def squash(self, x):
        """Squash the results using a softmax or sigmoid."""
        x = self.squash_layer(x)
        return x

    def zeros_and_ones(self, x, threshold=0.5):
        """Convert results to a hard 0 or 1."""
        x = self.squash(x)
        return (x > threshold).float()

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
