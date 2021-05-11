import torch.nn as nn
from torchvision.models.resnet import resnet34
import torch


class ResNet34(nn.Module):

    def __init__(self, num_classes: int = 250):
        super(ResNet34, self).__init__()
        self._backbone = resnet34(pretrained=True)
        self._backbone.fc = nn.Linear(self._backbone.fc.in_features, num_classes)
        self._fc2 = None
        self._fc3 = None
        self._relu = nn.ReLU()
        self._bn1d = None

    def adapt_fc(self):
        self._backbone.fc = nn.Linear(self._backbone.fc.in_features, 512)
        self._fc2 = nn.Linear(512, 512)
        self._fc3 = nn.Linear(512, 250)
        self._bn1d = nn.BatchNorm1d(512)

    def forward_extended(self, x):
        x = self._backbone(x)
        x = self._bn1d(x)
        x = self._relu(x)
        x = self._fc2(x)
        features = x
        normalized_features = features / torch.linalg.norm(features, keepdim=True, dim=1)
        x = self._relu(features)
        logits = self._fc3(x)
        return logits, normalized_features

    def forward(self, x):
        x = self._backbone(x)
        return x


class SiameseNetwork(nn.Module):

    def __init__(self, sketches_backbone: nn.Module, imagenet_backbone: nn.Module):
        super(SiameseNetwork, self).__init__()
        self._sketches_backbone = sketches_backbone
        self._imagenet_backbone = imagenet_backbone

    def forward(self, x, include_negative: bool = True):
        anchor, positive, negative = x
        anchor_logits, anchor_feats = self._sketches_backbone.forward_extended(anchor)  #
        positive_logits, positive_feats = self._imagenet_backbone.forward_extended(positive)

        if include_negative:
            negative_logits, negative_feats = self._imagenet_backbone.forward_extended(negative)
            return {'logits': dict(anchor=anchor_logits, positive=positive_logits, negative=negative_logits),
                    'feats': dict(anchor=anchor_feats, positive=positive_feats, negative=negative_feats)}
        return {'logits': dict(anchor=anchor_logits, positive=positive_logits),
                'feats': dict(anchor=anchor_feats, positive=positive_feats)}


if __name__ == '__main__':
    # load backbones
    print("Initializing weights...")
    imagenet_net = ResNet34()
    sketches_net = ResNet34()
    sketches_net.load_state_dict(torch.load('/home/rudy/Documents/cc7221/tarea2/weights/sketches.pth'))

    print("Adapting output layers...")
    sketches_net.adapt_fc()
    imagenet_net.adapt_fc()

    siamese_net = SiameseNetwork(sketches_net, imagenet_net)

    random_batch = torch.rand((8, 3, 224, 224))
    y = siamese_net((random_batch, random_batch, random_batch))


