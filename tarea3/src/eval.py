import torchvision


if __name__ == '__main__':
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=False,
        num_classes=10,
        pretrained_backbone=False,
        score_thresh=0.5,
        nms_thresh=0.5,
        detections_per_img=12
    )
    model.to('cuda')
