"""ResNet-18 adapted for CIFAR-10 (Papyan et al., 2020)."""

import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        base.maxpool = nn.Identity()

        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, nn.Flatten(),
        )
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))

    def features(self, x):
        return self.encoder(x)
