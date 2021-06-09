import json
import sys
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split

from bounding_box import BoundingBox
from datasets import OrandCarDataset, custom_collate_fn
from enums import *
from evaluator import get_coco_metrics
from models import create_model


class OCRTrainer(object):

    def __init__(self, model: nn.Module, dataset: OrandCarDataset, **kwargs):
        self.device = kwargs['device']
        self.epochs = kwargs['epochs']

        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs['initial_lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        self.dataset = dataset
        self.params = kwargs
        self.class_weight = float(kwargs['classification_weight'])
        self.regression_weight = 1 - self.class_weight

        # initialize data loaders
        n_val = int(len(self.dataset) * self.params['val_size'])
        n_train = len(self.dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=self.params['batch_size'], collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(val, batch_size=self.params['batch_size'] // 2, collate_fn=custom_collate_fn)

        # metrics
        self.metrics = defaultdict(list)  # epoch wise metrics

        self.use_wandb = kwargs['use_wandb']
        if self.use_wandb:
            wandb.init(project='homework1-cc7221', entity='p137')
            config = wandb.config
            config.model = self.model.__class__.__name__
            config.batch_size = kwargs['batch_size']
            config.epochs = self.epochs
            config.learning_rate = kwargs['initial_lr']
            config.optimizer = self.optimizer.__class__.__name__
            config.train_size = n_train
            config.val_size = n_val
            config.classification_weight = self.class_weight
            config.regression_weight = self.regression_weight

    def batch_to_device(self, batch: tuple):
        # move the batch to the device
        images = batch[0].to(self.device)
        targets = []
        for target in batch[1]:
            target['boxes'] = target['boxes'].to(self.device)
            target['labels'] = target['labels'].to(self.device)
            targets.append(target)
        return images, targets

    def save_metrics(self):
        path = f'metrics_{datetime.now()}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=4)
            if self.use_wandb:
                wandb.save(path)

    def optimize(self, batch: tuple) -> float:
        """
        Optimize the model for a given batch and a certain loss function using the optimizer defined at initialization.
        """

        self.optimizer.zero_grad()
        losses = self.model(*self.batch_to_device(batch))
        total_loss = losses['classification'] * self.class_weight + losses['bbox_regression'] * self.regression_weight
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def train(self):
        """
        Train a model. Evaluate once every epoch.
        """

        print("Training...")
        best_val_metric = 0
        for e in range(self.epochs):
            model.train()
            epoch_batch_loss = []
            for i, batch in enumerate(self.train_loader):

                batch_loss = self.optimize(batch)
                epoch_batch_loss.append(batch_loss)
                avg_epoch_loss = np.mean(epoch_batch_loss[-20:])  # use last 20 batch losses to calculate the mean

                sys.stdout.write('\r')
                sys.stdout.write(f"Epoch: {e + 1}({i + 1}/{len(self.train_loader)})| "
                                 f"Train loss: {avg_epoch_loss:.4f}")
                sys.stdout.flush()

                if self.use_wandb:
                    wandb.log({"train/loss": avg_epoch_loss})

            epoch_loss = np.mean(epoch_batch_loss)
            self.metrics['train_loss'].append(epoch_loss)

            checkpoint_metric = self.eval()
            self.lr_scheduler.step(checkpoint_metric)

            if checkpoint_metric > best_val_metric:
                print("Saving model...")
                best_val_metric = checkpoint_metric
                model_name = f"best_{self.model.__class__.__name__}.pth"
                torch.save(self.model.state_dict(), model_name)
                if self.use_wandb:
                    wandb.save(model_name)
        self.save_metrics()
        print("Training finished!")

    def eval(self) -> float:
        """
        Evaluate the current model. Return a single valued metric. This metric will be used for checkpointing.
        """
        self.model.eval()

        print("\nValidating")
        val_metrics = defaultdict(list)
        for batch in self.val_loader:
            preds = self.model(*self.batch_to_device(batch))
            gts, dts = self.to_bounding_box(batch[1], preds)

            iou_thresholds = np.linspace(0.5, 0.95, 10)
            for iou in iou_thresholds:
                # calculate map@50
                metrics = get_coco_metrics(gts, dts, iou_threshold=iou)  # metrics per class
                aps = [d['AP'] for d in metrics.values() if d['AP']]
                map_ = np.mean(aps) if len(aps) > 0 else 0
                val_metrics[f"iou_{iou}"].append(map_)

        map_5_95 = 0
        for iou_threshold in np.linspace(0.5, 0.95, 10):
            mean_map = np.mean(val_metrics[f"iou_{iou_threshold}"])
            map_5_95 += mean_map
            self.metrics[f'val_map_{iou_threshold}'].append(mean_map)
        map_5_95 /= 10
        map_5 = np.mean(val_metrics[f"iou_0.5"])

        # calculate mAP .5, .95
        if self.use_wandb:
            wandb.log({'val/map@.5': map_5, 'val/map@.5,.95': map_5_95})
        print(f"Val mAP@.5: {map_5:.2f}, mAP@.5,.95: {map_5_95:.2f}")
        return float(map_5_95)

    @staticmethod
    def to_bounding_box(gt_boxes: List[Dict[str, torch.Tensor]],
                        pred_boxes: List[Dict[str, torch.Tensor]]) -> Tuple[List[BoundingBox], List[BoundingBox]]:

        gt_parsed_boxes, pred_parsed_boxes = [], []

        # for each image
        for i, (gts, preds) in enumerate(zip(gt_boxes, pred_boxes)):
            image_name = str(i)

            # process ground truth bounding boxes
            boxes = list(gts['boxes'].detach().cpu().numpy())
            classes = list(gts['labels'].detach().cpu().numpy())
            for box, box_class in zip(boxes, classes):
                coords = tuple(box)
                b = BoundingBox(image_name=image_name,
                                class_id=box_class,
                                coordinates=coords,
                                type_coordinates=CoordinatesType.ABSOLUTE,
                                bb_type=BBType.GROUND_TRUTH,
                                format=BBFormat.XYX2Y2
                                )
                gt_parsed_boxes.append(b)

            # process detections
            boxes = list(preds['boxes'].detach().cpu().numpy())
            classes = list(preds['labels'].detach().cpu().numpy())
            conf = list(preds['scores'].detach().cpu().numpy())
            for box, box_class, box_confidence in zip(boxes, classes, conf):
                coords = tuple(box)
                b = BoundingBox(image_name=image_name,
                                class_id=box_class,
                                coordinates=coords,
                                confidence=box_confidence,
                                type_coordinates=CoordinatesType.ABSOLUTE,
                                bb_type=BBType.DETECTED,
                                format=BBFormat.XYX2Y2
                                )
                pred_parsed_boxes.append(b)
        return gt_parsed_boxes, pred_parsed_boxes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data/orand-car-with-bbs', type=str, help='Path to dataset folder')
    parser.add_argument('--val-size', default=0.05, type=float, help='Validation size')

    # model init
    parser.add_argument('--backbone', default='resnet50', help='Type of backbone')
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained model on COCO 2017')
    parser.add_argument('--pretrained-backbone', action='store_true',
                        help='Whether to use pretrained backbone on ImageNet')

    # model post-processing
    parser.add_argument('--score-thresh', default=0.5, type=float,
                        help="Score threshold used for postprocessing the detections.")
    parser.add_argument('--nms-thresh', default=0.5, type=float,
                        help="NMS threshold used for postprocessing the detections.")
    parser.add_argument('--detections-per-img', default=12, type=int,
                        help="Number of best detections to keep after NMS.")

    # training parameters
    parser.add_argument('--classification-weight', default=0.5, type=float,
                        help="Weight 'w' associated to classification loss."
                             "The regression weight will be determined"
                             "as 1 - 'w'")
    parser.add_argument('--batch-size', default=2, type=int, help="Batch size")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    model = create_model(args.backbone,
                         score_thresh=args.score_thresh,
                         nms_thresh=args.nms_thresh,
                         detections_per_img=args.detections_per_img)
    dataset = OrandCarDataset(args.data)
    trainer = OCRTrainer(model=model,
                         dataset=dataset,
                         epochs=args.epochs,
                         classification_weight=args.classification_weight,
                         initial_lr=args.lr,
                         batch_size=args.batch_size,
                         val_size=args.val_size,
                         device='cuda',
                         use_wandb=True)

    wandb.config.backbone = args.backbone
    trainer.train()
