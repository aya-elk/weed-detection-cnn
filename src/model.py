import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    (Conv -> BN -> ReLU) x2
    + optionnel MaxPool2d (downsampling)
    + optionnel Dropout2d
    """
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True, drop_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        if drop_p > 0:
            layers.append(nn.Dropout2d(drop_p))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class WeedCNN(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   16, pool=True,  drop_p=0.05),  # 256 -> 128
            ConvBlock(16,  32, pool=True,  drop_p=0.05),  # 128 -> 64
            ConvBlock(32,  64, pool=True,  drop_p=0.10),  # 64  -> 32
            ConvBlock(64, 128, pool=True,  drop_p=0.10),  # 32  -> 16
            ConvBlock(128,128, pool=False, drop_p=0.10),  # 16  -> 16
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
