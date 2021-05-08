import argparse
from collections import defaultdict
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from dataset import SketchTestDataset
from model_sketchz import ResNet34
from metrics import *
class Evaluator:

    def __init__(self, dataset: SketchTestDataset, true_labels: list, device: str = 'cuda', batch_size: int = 32):
        self.dataset = dataset
        self.true_labels = true_labels
        self.device = device
        self.batch_size = batch_size
        self.overall_accuracy = None
        self.test_acc_per_class = None
        self.model_accuracies = {}
#in contruction






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to folder containing test dataset')
    parser.add_argument('--device', default='cuda', type=str, help='Device in which to perform the evaluation')
    parser.add_argument('--weights', required=True, type=str, help='Path to weights of specified model')
    parser.add_argument('--output_image', default='models_acc.png', type=str,
                        help='Output path of per-class model results')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size used to evaluate model')

    args = parser.parse_args()
    torch.random.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = ResNet34(num_classes=8)

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
