import torch
import torch.nn as nn


class Resnet(nn.Module):
    """
    Resnet-34 adapted to fit CIFAR-10 problem. Key idea: "at the beginning of every layer
    the spacial dimensions are reduced to half and channels are duplicated". With this key idea
    we can adapt the dimensions of the CIFAR-10 images to work in a resnet-like architecture.
    """

    def __init__(self, n_classes):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(3, 16)
        self.layer2 = self._make_layer(4, 16, transition=True)
        self.layer3 = self._make_layer(6, 32, transition=True)
        self.layer4 = self._make_layer(3, 64, transition=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(128, n_classes)

    @staticmethod
    def _make_layer(n_blocks: int, channels: int, transition: bool = False):
        """
        If transition is true, the first block will downsample the spatial dimension to the half and
        double the amount of channels. The number of channels of the output is totally defined by
        the input and if there is a transition at the beginning of the layer.
        :param channels:            # input's channels of this layer.
        """
        blocks = [BasicBlock(channels, down_sample_space=transition, up_sample_channels=transition)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(channels * 2 if transition else channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        hidden = x
        x = self.dropout(x)
        x = self.fc(x)
        return {'hidden': hidden, 'logits': x}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, down_sample_space: bool = False, up_sample_channels: bool = False):
        super(BasicBlock, self).__init__()
        self.down_sample_space = down_sample_space
        out_channels = in_channels * 2 if up_sample_channels else in_channels

        if down_sample_space:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1))
            #  double the number of channels just at the beginning, now fix it
            in_channels = out_channels // 2 if up_sample_channels else in_channels
            self.skip_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1))
            self.skip_norm = nn.BatchNorm2d(out_channels)
        else:
            in_channels = out_channels / 2 if up_sample_channels else in_channels
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        if self.down_sample_space:
            identity = self.skip_conv(identity)
            identity = self.skip_norm(identity)

        out += identity
        out = self.relu(out)
        out = self.bn2(out)

        return out


if __name__ == '__main__':
    model = Resnet(19)