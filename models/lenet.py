# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["lenet"]


class LeNet(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.conv_2 = nn.Conv2d(6, 16, 5)

        if input_size == 32:
            self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        else:
            self.fc_1 = nn.Linear(16 * 4 * 4, 120)

        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv_2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


def lenet(num_classes, **kwargs):
    input_size = kwargs.get("input_size", 32)
    return LeNet(num_classes=num_classes, input_size=input_size)
