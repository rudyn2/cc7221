import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, residual=None, stride=(1, 1), cardinality=32):
        super(BottleneckBlock, self).__init__()
        self.expansion = 4
        self.base_width = 4
        width_ratio = out_channels / (self.expansion * 64)
        inter_channels = cardinality * int(self.base_width * width_ratio)

        self.conv1 = nn.Conv2d(
            in_channels, inter_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.conv2 = nn.Conv2d(inter_channels, inter_channels,
                               kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = nn.Conv2d(inter_channels, out_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.residual = residual
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.residual != None:
            identity = self.residual(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNext(nn.Module):
    def __init__(self, bottleneck_block, layers, image_channels, num_classes):
        super(ResNext, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Arquitectura ResNet

        self.layer1 = self._make_layer(
            bottleneck_block, layers[0], out_channels=256, stride=(1, 1)
        )
        self.layer2 = self._make_layer(
            bottleneck_block, layers[1], out_channels=512, stride=(2, 2)
        )
        self.layer3 = self._make_layer(
            bottleneck_block, layers[2], out_channels=1024, stride=(2, 2)
        )
        self.layer4 = self._make_layer(
            bottleneck_block, layers[3], out_channels=2048, stride=(2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #hidden = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        

        return x

    def _make_layer(self, bottleneck_block, num_residual_blocks, out_channels, stride):
        residual = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            residual = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(
            bottleneck_block(self.in_channels, out_channels, residual, stride)
        )

        self.in_channels = out_channels
        for i in range(num_residual_blocks - 1):
            layers.append(bottleneck_block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def resnext50(img_channel=3, num_classes=19):
    return ResNext(BottleneckBlock, [3, 4, 6, 3], img_channel, num_classes)
