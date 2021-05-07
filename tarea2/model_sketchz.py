import torch.nn as nn
from torchvision.models.resnet import  resnet34


class ResNet34(nn.Module):

    def __init__(self, num_classes: int = 250):
        super(ResNet34, self).__init__()
        self._backbone = resnet34(pretrained=True)
        self._backbone.fc = nn.Linear(self._backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self._backbone(x)
        return x