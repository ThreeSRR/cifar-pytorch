# -*-coding:utf-8-*-
import torch.nn as nn

__all__ = ["alexnet"]


class AlexNet(nn.Module):
    def __init__(self, kernel_sizes, strides, paddings, channels, num_classes):
        super(AlexNet, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=paddings[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_sizes[1], stride=strides[1], padding=paddings[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=kernel_sizes[2], stride=strides[2], padding=paddings[2]),
            nn.ReLU(inplace=True)
            )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=kernel_sizes[3], stride=strides[3], padding=paddings[3]),
            nn.ReLU(inplace=True)
            )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], kernel_size=kernel_sizes[4], stride=strides[4], padding=paddings[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(channels[4], num_classes)

    def forward(self, x):
        # print('input shape:', x.shape)
        x = self.layer_1(x)
        # print('layer_1 shape:', x.shape)
        x = self.layer_2(x)
        # print('layer_2 shape:', x.shape)
        x = self.layer_3(x)
        # print('layer_3 shape:', x.shape)
        x = self.layer_4(x)
        # print('layer_4 shape:', x.shape)
        x = self.layer_5(x)
        # print('layer_5 shape:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # exit()
        return x


def alexnet(num_classes, **kwargs):

    channels = [64, 192, 384, 256, 256]

    if 'kernel_sizes' in kwargs:
        kernel_sizes_setting = kwargs['kernel_sizes']
        print('change kernel_sizes, using default strides')
        strides = [4, 1, 1, 1, 1]

        cfg = kernel_size_config(kernel_sizes_setting)
        kernel_sizes = cfg['kernel_sizes']
        paddings = cfg['paddings']

    elif 'channels' in kwargs:
        channels = kwargs['channels']
        channels = [int(i) for i in channels.split('-')]
        print('change channels, using default kernel_sizes and strides')
        kernel_sizes = [11, 5, 3, 3, 3]
        strides = [4, 1, 1, 1, 1]
        paddings = [5, 2, 1, 1, 1]

    else:
        feature_size_setting = kwargs['feature_size']
        print('change feature_size, using default kernel_sizes')
        kernel_sizes = [11, 5, 3, 3, 3]

        cfg = feature_size_config(feature_size_setting)
        strides = cfg['strides']
        paddings = cfg['paddings']
        
    
    print('kernel_sizes:', kernel_sizes, 'strides:', strides, 'paddings:', paddings)
    return AlexNet(kernel_sizes=kernel_sizes, strides=strides, paddings=paddings, channels=channels, num_classes=num_classes)


def kernel_size_config(kernel_sizes_setting):
    cfg = {
        '11-5-3-3-3': {
            'kernel_sizes': [11, 5, 3, 3, 3],
            'paddings': [5, 2, 1, 1, 1]
        },
        '7-5-3-3-3': {
            'kernel_sizes': [7, 5, 3, 3, 3],
            'paddings': [3, 2, 1, 1, 1]
        },
        '5-5-3-3-3': {
            'kernel_sizes': [5, 5, 3, 3, 3],
            'paddings': [2, 2, 1, 1, 1]
        },
        '7-7-3-3-3': {
            'kernel_sizes': [7, 7, 3, 3, 3],
            'paddings': [3, 3, 1, 1, 1]
        },
        '7-3-3-3-3': {
            'kernel_sizes': [7, 3, 3, 3, 3],
            'paddings': [3, 1, 1, 1, 1]
        },
        '5-3-3-3-3': {
            'kernel_sizes': [5, 3, 3, 3, 3],
            'paddings': [2, 1, 1, 1, 1]
        },
        '3-3-3-3-3': {
            'kernel_sizes': [3, 3, 3, 3, 3],
            'paddings': [1, 1, 1, 1, 1]
        },
        '7-7-5-3-3': {
            'kernel_sizes': [7, 7, 5, 3, 3],
            'paddings': [3, 3, 2, 1, 1]
        },
        '7-7-5-5-5': {
            'kernel_sizes': [7, 7, 5, 5, 5],
            'paddings': [3, 3, 2, 2, 2]
        },
        '3-3-5-7-7': {
            'kernel_sizes': [3, 3, 5, 7, 7],
            'paddings': [1, 1, 2, 3, 3]
        },
        '3-3-5-5-5': {
            'kernel_sizes': [3, 3, 5, 5, 5],
            'paddings': [1, 1, 2, 2, 2]
        },
        '3-3-3-5-7': {
            'kernel_sizes': [3, 3, 3, 5, 7],
            'paddings': [1, 1, 1, 2, 3]
        },
        '3-3-3-5-11': {
            'kernel_sizes': [3, 3, 3, 5, 11],
            'paddings': [1, 1, 1, 2, 5]
        },
        '3-3-3-5-5': {
            'kernel_sizes': [3, 3, 3, 5, 5],
            'paddings': [1, 1, 1, 2, 2]
        },
        '3-3-3-3-5': {
            'kernel_sizes': [3, 3, 3, 3, 5],
            'paddings': [1, 1, 1, 1, 2]
        },
        '3-3-3-3-7': {
            'kernel_sizes': [3, 3, 3, 3, 7],
            'paddings': [1, 1, 1, 1, 3]
        }
    }

    return cfg[kernel_sizes_setting]


def feature_size_config(feature_size_setting):

    cfg = {
        '4-2-2-2-1': {
            'strides': [4, 1, 1, 1, 1],
            'paddings': [5, 2, 1, 1, 1]
        },
        '8-4-2-2-1': {
            'strides': [2, 1, 1, 1, 1],
            'paddings': [5, 2, 0, 1, 1]
        },
        '8-4-4-2-1': {
            'strides': [2, 1, 1, 1, 1],
            'paddings': [5, 2, 1, 0, 1]
        },
        '8-4-4-4-1': {
            'strides': [2, 1, 1, 1, 1],
            'paddings': [5, 2, 1, 1, 0]
        },
        '16-8-4-4-1': {
            'strides': [1, 1, 2, 1, 1],
            'paddings': [5, 2, 1, 1, 0]
        },
        '16-8-8-4-1': {
            'strides': [1, 1, 1, 2, 1],
            'paddings': [5, 2, 1, 1, 0]
        },
        '16-8-4-2-1': {
            'strides': [1, 1, 2, 1, 1],
            'paddings': [5, 2, 1, 0, 1]
        }
    }

    return cfg[feature_size_setting]
