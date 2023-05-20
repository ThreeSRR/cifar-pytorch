# -*-coding:utf-8-*-
import torch.nn as nn

__all__ = ["vgg11", "vgg13", "vgg16", "vgg19"]

cfg = {
  "A": [[64, 3, 1, 1], "M", [128, 3, 1, 1], "M", [256, 3, 1, 1], [256, 3, 1, 1], "M", [512, 3, 1, 1], [512, 3, 1, 1], "M", [512, 3, 1, 1], [512, 3, 1, 1], "M"],
  "B": [[64, 3, 1, 1], [64, 3, 1, 1], "M", [128, 3, 1, 1], [128, 3, 1, 1], "M", [256, 3, 1, 1], [256, 3, 1, 1], "M", [512, 3, 1, 1], [512, 3, 1, 1], "M", [512, 3, 1, 1], [512, 3, 1, 1], "M"],
  
  "D": [[64, 7, 1, 3], [64, 7, 1, 3], "M", # 32x32 -> 16x16
        [128, 3, 1, 1], [128, 3, 1, 1], "M", # 16x16 -> 8x8
        [256, 3, 1, 1], [256, 3, 1, 1], [256, 3, 1, 1], "M", # 8x8 -> 4x4
        [512, 3, 1, 1], [512, 3, 1, 1], [512, 3, 1, 1], "M", #4x4 -> 2x2
        [512, 3, 1, 1], [512, 3, 1, 1], [512, 3, 1, 1], "M"], #2x2 -> 1x1
  "E": [[64, 3, 1, 1], [64, 3, 1, 1], "M", 
        [128, 3, 1, 1], [128, 3, 1, 1], "M", 
        [256, 3, 1, 1], [256, 3, 1, 1], [256, 3, 1, 1], [256, 3, 1, 1], "M", 
        [512, 3, 1, 1], [512, 3, 1, 1], [512, 3, 1, 1], [512, 3, 1, 1], "M", 
        [512, 3, 1, 1], [512, 3, 1, 1], [512, 3, 1, 1], [512, 3, 1, 1], "M"]
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        #self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def vgg11(num_classes, **kwargs):
    return VGG(make_layers(cfg["A"], batch_norm=True), num_classes)


def vgg13(num_classes, **kwargs):
    return VGG(make_layers(cfg["B"], batch_norm=True), num_classes)


def vgg16(num_classes, **kwargs):
    return VGG(make_layers(cfg["D"], batch_norm=True), num_classes)


def vgg19(num_classes, **kwargs):
    return VGG(make_layers(cfg["E"], batch_norm=True), num_classes)