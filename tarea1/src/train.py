from dataset import ImageDataset
from torch.utils.data import DataLoader

from resnet import Resnet
import wandb

import argparse
import logging
import os
import sys
import torch.nn.functional as F

import torch
import time
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import defaultdict


def train_for_classification(net, dataset, optimizer,
                             criterion, lr_scheduler=None,
                             epochs=1, reports_every=1, device='cuda', val_percent: float = 0.1):
    net.to(device)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    tiempo_epochs = 0
    train_loss, train_acc, test_acc = [], [], []

    for e in range(1, epochs + 1):
        inicio_epoch = time.time()
        net.train()

        # Variables para las métricas
        running_loss, running_acc = 0.0, 0.0
        avg_acc, avg_loss = 0, 0

        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # optimization step
            optimizer.zero_grad()
            out_dict = net(images.float())
            y_pred = out_dict['logits'].type(torch.DoubleTensor)  # probability distribution over classes
            labels = labels.type(torch.LongTensor)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            # loss
            items = min(n_train, (i + 1) * train_loader.batch_size)
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            # accuracy
            _, max_idx = torch.max(y_pred, dim=1)
            running_acc += torch.sum(max_idx == labels).item()
            avg_acc = running_acc / items * 100

            # report
            sys.stdout.write(f'\rEpoch:{e}({items}/{n_train}), '
                             + (f'lr:{lr_scheduler.get_last_lr()[0]:02.7f}, ' if lr_scheduler is not None else '')
                             + f'Loss:{avg_loss:02.5f}, '
                             + f'Train Acc:{avg_acc:02.1f}%')
            wandb.log({'train/loss': avg_loss, 'train/acc': train_acc})

        tiempo_epochs += time.time() - inicio_epoch

        if e % reports_every == 0:
            sys.stdout.write(', Validating...')

            train_loss.append(avg_loss)
            train_acc.append(avg_acc)

            avg_acc = eval_net(device, net, val_loader)
            test_acc.append(avg_acc)
            sys.stdout.write(f', Val Acc:{avg_acc:02.2f}%, '
                             + f'Avg-Time:{tiempo_epochs / e:.3f}s.\n')
            wandb.log({'val/loss': avg_loss, 'val/acc': train_acc})
        else:
            sys.stdout.write('\n')

        if lr_scheduler is not None:
            lr_scheduler.step()

    return train_loss, (train_acc, test_acc)


def eval_net(device, net, test_loader):
    net.eval()
    running_acc = 0.0
    total_test = 0

    for i, data in enumerate(test_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        out_dict = net(images)
        logits = out_dict['logits']
        _, max_idx = torch.max(logits, dim=1)
        running_acc += torch.sum(max_idx == labels).item()
        total_test += len(labels)

    avg_acc = (running_acc / total_test) * 100
    return avg_acc


if __name__ == '__main__':
    wandb.init(project='homework1-cc7221', entity='p137')
    config = wandb.config
    config.learning_rate = 0.01

    torch.cuda.empty_cache()
    train_dataset = ImageDataset(r"C:\Users\C0101\PycharmProjects\cc7221\data\clothing-small", 224, 224)
    backbone_resnet = Resnet(19)
    optimizer = torch.optim.Adam(backbone_resnet.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss()

    train_for_classification(backbone_resnet,
                             train_dataset,
                             optimizer,
                             criterion)
