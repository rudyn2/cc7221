import argparse
from collections import defaultdict
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from alexnet import AlexNet
from dataset import TestImageDataset
from resnet50 import ResNet50
from resnext50 import resnext50


class Evaluator:

    def __init__(self, dataset: TestImageDataset, true_labels: list, device: str = 'cuda', batch_size: int = 32):
        self.dataset = dataset
        self.true_labels = true_labels
        self.device = device
        self.batch_size = batch_size
        self.overall_accuracy = None
        self.test_acc_per_class = None
        self.model_accuracies = {}

    def get_accuracy(self, net: torch.nn.Module, ) -> Tuple[float, dict]:
        """
        Calculates overall accuracy and accuracy per class
        """
        test_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        overall_accuracy = 0

        class_count = defaultdict(int)
        class_tp_count = defaultdict(int)

        for i, data in tqdm(enumerate(test_loader), f"Evaluating {net.__class__.__name__} ...", total=len(test_loader)):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

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
        overall_accuracy /= len(self.dataset)
        per_class_acc = {}
        for class_label in true_labels:
            per_class_acc[class_label] = class_tp_count[class_label] / class_count[class_label]

        self.overall_accuracy = overall_accuracy
        self.test_acc_per_class = per_class_acc
        self.model_accuracies[net.__class__.__name__] = dict(mean_acc=overall_accuracy, class_acc=per_class_acc)
        return overall_accuracy, per_class_acc

    def calculate_models_accuracy(self, models: List[torch.nn.Module]):
        for m in models:
            self.get_accuracy(m)

    def plot_class_accuracy(self, models: List[torch.nn.Module], labels: list, save_path: str):

        # TODO: Make this code generalize for any number of models
        assert len(models) == 3

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects = []
        for i, m in enumerate(models):
            name = m.__class__.__name__
            values = self.model_accuracies[name]['class_acc'].values()
            rect = ax.bar(x - width * np.floor(len(models) / 2) + i * width, values, width, label=name)
            rects.append(rect)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy [%]')
        ax.set_xlabel('Clase')
        ax.set_title('Accuracy por clase y modelo')
        ax.set_xticks(x)
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.show()

    def print_results(self):
        for m in self.model_accuracies.keys():
            print(f"Model: {m}, Test accuracy: {self.model_accuracies[m]['mean_acc']:.3f}")


def select_model_by_name(name: str):
    # model selection
    if name == 'resnet':
        model = ResNet50(img_channel=3, num_classes=19)
        # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
    elif name == 'resnet_torch':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
    elif name == 'resnext':
        model = resnext50(img_channel=3, num_classes=19)
    elif name == 'alexnet':
        model = AlexNet(num_classes=19)
    else:
        raise ValueError("This utility can't train that kind of model.")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to folder containing test dataset')
    parser.add_argument('--models', default='resnet', type=str, help='Type of model (resnet, resnext, alexnet)')
    parser.add_argument('--device', default='cuda', type=str, help='Device in which to perform the evaluation')
    parser.add_argument('--weights', required=True, type=str, help='Path to weights of specified model')
    parser.add_argument('--output_image', default='models_acc.png', type=str,
                        help='Output path of per-class model results')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size used to evaluate model')

    args = parser.parse_args()
    torch.random.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    models = []
    if not args.models.__contains__(','):
        models.append(select_model_by_name(args.models))
    else:
        model_names = args.models.split(",")
        for model_name in model_names:
            models.append(select_model_by_name(model_name))

    # load weights
    if not args.weights.__contains__(','):
        models[0].load_state_dict(torch.load(args.weights))
        models[0].to(args.device)
        models[0].eval()
    else:
        print("Multiple models...")
        weight_paths = args.weights.split(",")
        for i, weight_path in enumerate(weight_paths):
            print(f"Loading {weight_path}")
            models[i].load_state_dict(torch.load(weight_path))
            models[i].to(args.device)
            models[i].eval()

    # create test dataset
    test_dataset = TestImageDataset(args.data, 224, 224)
    true_labels = list(range(19))
    xtick_labels = list(test_dataset.read_mapping().values())

    # evaluate
    e = Evaluator(test_dataset, true_labels, batch_size=args.batch_size)
    e.calculate_models_accuracy(models)
    e.plot_class_accuracy(models, xtick_labels, args.output_image)
    e.print_results()
