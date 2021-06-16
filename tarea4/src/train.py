import json
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.models.segmentation
import wandb
from torch.utils.data import DataLoader, random_split
from tarea4.src.datasets import SpermDataset
from tarea4.src.losses import FocalLoss


class SpermTrainer(object):

    def __init__(self, model: nn.Module, dataset: SpermDataset, **kwargs):
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
        self.train_loader = DataLoader(train, batch_size=self.params['batch_size'])
        self.val_loader = DataLoader(val, batch_size=self.params['batch_size'] // 2)

        # metrics
        self.metrics = defaultdict(list)  # epoch wise metrics

        # loss
        self.loss = nn.CrossEntropyLoss()

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
        images = batch[0]
        mask = batch[1]
        self.optimizer.zero_grad()
        pred_mask = self.model(images)['out']
        total_loss = self.loss(pred_mask, mask)
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
            break

        return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data/SpermSegGS', type=str, help='Path to dataset folder')
    parser.add_argument('--val-size', default=0.05, type=float, help='Validation size')

    # training parameters
    parser.add_argument('--batch-size', default=2, type=int, help="Batch size")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=3, pretrained=False)
    dataset = SpermDataset(args.data)
    trainer = SpermTrainer(model=model,
                           dataset=dataset,
                           epochs=args.epochs,
                           initial_lr=args.lr,
                           batch_size=args.batch_size,
                           val_size=args.val_size,
                           device='cuda',
                           use_wandb=False)

    trainer.train()
