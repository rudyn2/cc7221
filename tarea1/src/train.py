from dataset import ImageDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch
from resnet import Resnet
import wandb


class ImageClassifier(pl.LightningModule):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        y_predicted = self.backbone(x)
        return y_predicted

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_pred = self.backbone(x)['logits'].type(torch.DoubleTensor)        # probability distribution over classes
        y = y.type(torch.LongTensor)                                        # encoded class label as integer
        loss = self.loss(y_pred, y)
        wandb.log({'loss': loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    wandb.init(project='homework1-cc7221', entity='p137')
    config = wandb.config
    config.learning_rate = 0.01

    torch.cuda.empty_cache()
    train_dataset = ImageDataset(r"C:\Users\C0101\PycharmProjects\cc7221\data\clothing-small", 224, 224)
    train_dataloader = DataLoader(train_dataset, batch_size=8, pin_memory=True, shuffle=True)
    backbone_resnet = Resnet(19)
    classifier = ImageClassifier(backbone_resnet)
    classifier.to(torch.double)
    trainer = pl.Trainer(gpus=0, precision=32, limit_train_batches=0.5)
    trainer.fit(classifier, train_dataloader)

    # single_example, _ = train_dataset[0]
    #
    # for images, labels in train_dataloader:
    #     y = backbone_resnet(images.float())
    #     break
