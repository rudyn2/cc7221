from abc import ABC
from abc import abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import OrandCarDataset, custom_collate_fn
import sys
from collections import defaultdict
from utils import calculate_map
import numpy as np
import wandb
from evaluator import get_coco_metrics


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

        # initialize data loaders
        n_val = int(len(self.dataset) * self.params['val_size'])
        n_train = len(self.dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=self.params['batch_size'], collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(val, batch_size=self.params['batch_size'] // 2, collate_fn=custom_collate_fn)

        # metrics
        self.metrics = defaultdict(list)

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

    def batch_to_device(self, batch: tuple):
        # move the batch to the device
        images = batch[0].to(self.device)
        targets = []
        for target in batch[1]:
            target['boxes'] = target['boxes'].to(self.device)
            target['labels'] = target['labels'].to(self.device)
            targets.append(target)
        return images, targets

    def optimize(self, batch: tuple):

        self.optimizer.zero_grad()
        losses = self.model(*self.batch_to_device(batch))
        total_loss = losses['classification'] * 0.5 + losses['bbox_regression'] * 0.5
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def train(self):

        print("Training...")
        best_val_metric = 0
        for e in range(self.epochs):
            model.train()
            epoch_batch_loss = []
            for i, batch in enumerate(self.train_loader):

                batch_loss = self.optimize(batch)
                epoch_batch_loss.append(batch_loss)
                avg_epoch_loss = np.mean(epoch_batch_loss[-20:])    # use last 20 batch losses to calculate the mean

                sys.stdout.write('\r')
                sys.stdout.write(f"Epoch: {e + 1}({i + 1}/{len(self.train_loader)})| "
                                 f"Train loss: {avg_epoch_loss:.4f}")
                sys.stdout.flush()

                if self.use_wandb:
                    wandb.log({"train/loss": avg_epoch_loss})

            epoch_loss = np.mean(epoch_batch_loss)
            # if self.use_wandb:
            #     wandb.log({'train/loss': epoch_loss})
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

    def eval(self):
        """
        Evaluate the current model. Return a single valued metric. This metric will be used for checkpointing.
        """
        self.model.eval()
        val_maps_50, val_maps_95 = [], []

        print("\nValidating")
        for batch in self.val_loader:
            preds = self.model(*self.batch_to_device(batch))
            gts, dts = self.to_bounding_box(batch[1], preds)

            # calculate map@50
            metrics = get_coco_metrics(gts, dts, iou_threshold=0.5)    # metrics per class
            aps_50 = [d['AP'] for d in metrics.values() if d['AP']]
            map_50 = np.mean(aps_50) if len(aps_50) > 0 else 0
            val_maps_50.append(map_50)

            # calculate map@95
            metrics = get_coco_metrics(gts, dts, iou_threshold=0.95)    # metrics per class
            aps_95 = [d['AP'] for d in metrics.values() if d['AP']]
            map_95 = np.mean(aps_95) if len(aps_95) > 0 else 0
            val_maps_95.append(map_95)

        mean_map_50 = np.mean(val_maps_50)
        mean_map_95 = np.mean(val_maps_95)
        self.metrics['val_map_50'].append(mean_map_50)

        if self.use_wandb:
            wandb.log({'val/map@50': mean_map_50, 'val/map@95': mean_map_95})
        print(f"Val mAP@50: {mean_map_50:.2f}, mAP@95: {mean_map_95:.2f}")
        return mean_map_50

    @staticmethod
    def to_bounding_box(gt_boxes, pred_boxes):
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
    import torchvision
    from bounding_box import BoundingBox
    from enums import *
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=False,
        num_classes=10,
        pretrained_backbone=False
    )
    dataset = OrandCarDataset("/home/rudy/Documents/cc7221/tarea3/data/orand-car-with-bbs")
    trainer = OCRTrainer(model=model,
                         dataset=dataset,
                         epochs=10,
                         initial_lr=0.0001,
                         batch_size=4,
                         val_size=0.06,
                         device='cuda',
                         use_wandb=True)
    trainer.train()



