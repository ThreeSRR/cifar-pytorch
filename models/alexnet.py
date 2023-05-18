# -*-coding:utf-8-*-
import torch.nn as nn

__all__ = ["alexnet"]


class AlexNet(nn.Module):
    def __init__(self, kernel_sizes, strides, paddings, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=kernel_sizes[1], stride=strides[1], padding=paddings[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=kernel_sizes[2], stride=strides[2], padding=paddings[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=kernel_sizes[3], stride=strides[3], padding=paddings[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=kernel_sizes[4], stride=strides[4], padding=paddings[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def alexnet(num_classes, kernel_sizes=[11, 5, 3, 3, 3], strides=[4, 1, 1, 1, 1], paddings=[5, 2, 1, 1, 1], **kwargs):
    assert len(kernel_sizes) == 5 and len(strides) == 5 and len(paddings) == 5, \
        "kernel_sizes must be a list of 5 elements"
    return AlexNet(kernel_sizes=kernel_sizes, strides=strides, paddings=paddings, num_classes=num_classes)
