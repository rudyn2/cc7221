import torch.nn as nn
from torchvision.models.resnet import resnet34
import torch


class ResNet34(nn.Module):

    def __init__(self, num_classes: int = 250):
        super(ResNet34, self).__init__()
        self.backbone = resnet34(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self._fc2 = None
        self._fc3 = None
        self._bn1d = None
        self._relu = nn.ReLU()

    def adapt_fc(self, fc1: nn.Module, fc2: nn.Module, fc3: nn.Module, bn: nn.Module):
        self.backbone.fc = fc1
        self._fc2 = fc2
        self._fc3 = fc3
        self._bn1d = bn

    def forward_extended(self, x):
        x = self.backbone(x)
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

    def __init__(self, sketches_backbone: ResNet34, imagenet_backbone: ResNet34):
        super(SiameseNetwork, self).__init__()
        self._sketches_side = sketches_backbone
        self._imagenet_side = imagenet_backbone
        self._backbone_output_dim = self._sketches_side.backbone.fc.in_features

        self.share_parameters() # create shared parameters and assign them

    def share_parameters(self):
        fc1 = nn.Linear(self._backbone_output_dim, 512)
        fc2 = nn.Linear(512, 512)
        fc3 = nn.Linear(512, 250)
        bn = nn.BatchNorm1d(512)

        # both have same fc layers
        self._sketches_side.adapt_fc(fc1, fc2, fc3, bn)
        self._imagenet_side.adapt_fc(fc1, fc2, fc3, bn)

    def extract_features_image(self, image_batch):
        _, feats = self._imagenet_side.forward_extended(image_batch)
        return feats

    def extract_features_sketch(self, image_batch):
        _, feats = self._sketches_side.forward_extended(image_batch)
        return feats

    def forward(self, x, include_negative: bool = True):
        anchor, positive, negative = x
        anchor_logits, anchor_feats = self._sketches_side.forward_extended(anchor)
        positive_logits, positive_feats = self._imagenet_side.forward_extended(positive)

        if include_negative:
            negative_logits, negative_feats = self._imagenet_side.forward_extended(negative)
            return {'logits': dict(anchor=anchor_logits, positive=positive_logits, negative=negative_logits),
                    'feats': dict(anchor=anchor_feats, positive=positive_feats, negative=negative_feats)}
        return {'logits': dict(anchor=anchor_logits, positive=positive_logits),
                'feats': dict(anchor=anchor_feats, positive=positive_feats)}


if __name__ == '__main__':
    # load backbones
    print("Initializing weights...")
    imagenet_net = ResNet34()
    sketches_net = ResNet34()
    # sketches_net.load_state_dict(torch.load('/home/rudy/Documents/cc7221/tarea2/weights/sketches.pth'))

    print("Adapting output layers...")
    siamese_net = SiameseNetwork(sketches_net, imagenet_net)

    random_batch = torch.rand((8, 3, 224, 224))
    y = siamese_net((random_batch, random_batch, random_batch))
