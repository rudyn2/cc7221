import argparse
from collections import defaultdict
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from alexnet import AlexNet
from dataset import TestImageDataset
from resnet50 import ResNet50
from resnext50 import resnext50


def get_accuracy(net: torch.nn.Module, dataset: TestImageDataset, true_labels: list, device: str = 'cuda',
                 batch_size: int = 32) -> Tuple[float, dict]:
    """
    Calculates overall accuracy and accuracy per class
    """
    test_loader = DataLoader(dataset, batch_size=batch_size)
    overall_accuracy = 0

    class_count = defaultdict(int)
    class_tp_count = defaultdict(int)

    for i, data in tqdm(enumerate(test_loader), "Evaluating...", total=len(test_loader)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # optimization step
        logits = net(images)
        y_pred = logits.type(torch.DoubleTensor)  # probability distribution over classes
        labels = labels.type(torch.LongTensor)
        _, max_idx = torch.max(y_pred, dim=1)
        batch_tp = torch.sum(max_idx == labels).item()

        for label in true_labels:
            mask = labels == label
            class_labels = labels[mask]
            class_pred = y_pred[mask]

            if class_labels.shape[0] > 0:
                if len(class_pred.shape) > 1:
                    _, class_max_idx = torch.max(class_pred, dim=1)
                else:
                    _, class_max_idx = torch.max(class_pred, dim=0)
                class_tp = torch.sum(class_max_idx == class_labels).item()
                class_count[label] += len(class_labels)
                class_tp_count[label] += class_tp

        overall_accuracy += batch_tp

    # summarize metrics
    overall_accuracy /= len(dataset)
    per_class_acc = {}
    for class_label in true_labels:
        per_class_acc[class_label] = class_tp_count[class_label] / class_count[class_label]

    return overall_accuracy, per_class_acc


def plot_class_accuracy(class_acc: dict):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to folder containing test dataset')
    parser.add_argument('--model', default='resnet', type=str, help='Type of model (resnet, resnext, alexnet)')
    parser.add_argument('--device', default='cuda', type=str, help='Device in which to perform the evaluation')
    parser.add_argument('--weights', required=True, type=str, help='Path to weights of specified model')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size used to evaluate model')

    args = parser.parse_args()

    # model selection
    if args.model == 'resnet':
        model = ResNet50(img_channel=3, num_classes=19)
        # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
    elif args.model == 'resnet_torch':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
    elif args.model == 'resnext':
        model = resnext50(img_channel=3, num_classes=19)
    elif args.model == 'alexnet':
        model = AlexNet(num_classes=19)
    else:
        raise ValueError("This utility can't train that kind of model.")

    # load weights
    model.load_state_dict(torch.load(args.weights))
    model.to(args.device)
    model.eval()

    # create test dataset
    test_dataset = TestImageDataset(args.data, 224, 224)
    true_labels = list(range(19))
    test_acc, test_per_class_acc = get_accuracy(model, test_dataset, true_labels, device=args.device,
                                                batch_size=args.batch_size)
    print(f"Test accuracy: {test_acc:.3f}")
