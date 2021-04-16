import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(
            self, in_channels, inter_channels, residual=None, stride=(1, 1)):
        super(ResBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, inter_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels,
                               kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, inter_channels * self.expansion,
                               kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(inter_channels * self.expansion)
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


class ResNet(nn.Module):
    def __init__(self, Res_block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Arquitectura ResNet

        self.layer1 = self._make_layer(Res_block, layers[0], inter_channels=64, stride=(1, 1))
        self.layer2 = self._make_layer(Res_block, layers[1], inter_channels=128, stride=(2, 2))
        self.layer3 = self._make_layer(Res_block, layers[2], inter_channels=256, stride=(2, 2))
        self.layer4 = self._make_layer(Res_block, layers[3], inter_channels=512, stride=(2, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # hidden = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, Res_block, num_residual_blocks, inter_channels, stride):
        residual = None
        layers = []

        # si reducimos a la mitad el espacio de entrada, 56x56 -> 28x28 (stride=2), o
        # los canales cambian necesitamos adaptar la identidad para que se pueda agregar a una capa posterior
        if stride != 1 or self.in_channels != inter_channels * 4:
            residual = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    inter_channels * 4,
                    kernel_size=(1, 1),
                    stride=stride,
                ),
                nn.BatchNorm2d(inter_channels * 4),
            )

        layers.append(
            Res_block(self.in_channels, inter_channels, residual, stride)
        )

        # la expansion
        self.in_channels = inter_channels * 4

        # Se agregan los bloques necesarios
        for i in range(num_residual_blocks - 1):
            layers.append(Res_block(self.in_channels, inter_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=19):
    return ResNet(ResBlock, [3, 4, 6, 3], img_channel, num_classes)


if __name__ == '__main__':
    x = torch.rand((64, 3, 224, 224))
    x = x.to('cuda')
    m = ResNet50()
    m.to('cuda')
    y = m(x)
