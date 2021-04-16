import torch
import wandb
import torch.nn as nn
import argparse
import logging

from dataset import TrainImageDataset
from resnet50 import ResNet50
from resnext50 import resnext50
from alexnet import AlexNet
from train import train_for_classification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', required=True, type=str, help='Path to folder containing dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--model', default='resnet', type=str, help='Type of model (resnet, resnext, alexnet)')

    args = parser.parse_args()

    wandb.init(project='homework1-cc7221', entity='p137')

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

    logging.info("Setting dataset")
    torch.cuda.empty_cache()
    train_dataset = TrainImageDataset(args.data, 224, 224)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    config = wandb.config

    # wandb config specification
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.model = args.model

    logging.info("Training...")
    train_for_classification(net=model, dataset=train_dataset, batch_size=args.batch_size,
                             optimizer=optimizer, criterion=criterion, epochs=args.epochs)

    logging.info("Saving...")
    model_name = f"{args.model}_{args.epochs}.pth"
    torch.save(model.state_dict(), model_name)
    wandb.save(model_name)
