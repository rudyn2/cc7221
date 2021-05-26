from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def create_model(backbone_name: str, num_classes: int = 10, **kwargs):
    """
    backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
    """
    backbone = resnet_fpn_backbone(backbone_name, False,
                                   returned_layers=[2, 3, 4],
                                   extra_blocks=LastLevelP6P7(256, 256),
                                   trainable_layers=5)
    return RetinaNet(backbone, num_classes, **kwargs)
