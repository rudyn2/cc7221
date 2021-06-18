import torch
from datasets import SpermDataset
from termcolor import colored
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, ConfusionMatrix, DiceCoefficient, IoU, MetricsLambda
from ignite.utils import to_onehot
from ignite.contrib.handlers import WandBLogger
from losses import FocalLoss
from tqdm import tqdm
import torchvision


def prepare_batch(batch, device, non_blocking):
    x = batch[0].to(device, non_blocking=non_blocking)
    y = batch[1].to(device, non_blocking=non_blocking)
    y = y.squeeze(1)
    return x, y


def output_transform_seg(process_output):
    """
    Output transform for segmentation metrics.
    """

    y_pred = process_output[0]['out'].argmax(dim=1)  # (B, W, H)
    y = process_output[1]  # (B, W, H)
    y_pred_ = y_pred.view(-1)  # B, (W*H)
    y_ = y.view(-1)
    y_pred_one_hot = to_onehot(y_pred_, num_classes=3)
    return dict(y_pred=y_pred_one_hot, y=y_)  # output format is according to `DiceCoefficient` docs


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(colored("Using device: ", "white") + colored(device, "green"))

    print(colored("[*] Initializing dataset and dataloader", "white"))
    dataset = SpermDataset(args.data)
    n_val = int(len(dataset) * 0.05)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(colored("[+] Dataset & Dataloader Ready!", "green"))

    print(colored("[*] Initializing model, optimizer and loss", "white"))
    # model = DLv3Wrapper()
    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=3, pretrained=False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = FocalLoss(apply_nonlin=torch.sigmoid)
    print(colored("[+] Model, optimizer and loss are ready!", "green"))

    avg_fn = lambda x: torch.mean(x).item()
    cm_metric = ConfusionMatrix(num_classes=3, output_transform=output_transform_seg)
    metrics = {'loss': Loss(loss_fn=loss, output_transform=lambda x: (x[0], x[1])),
               'dice': MetricsLambda(avg_fn, DiceCoefficient(cm_metric)),
               'iou': MetricsLambda(avg_fn, IoU(cm_metric))}
    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        loss,
                                        prepare_batch=prepare_batch,
                                        output_transform=lambda x, y, y_, l: (y_, y, l),
                                        device=device)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler=lambda _: train_evaluator.run(train_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler=lambda _: val_evaluator.run(val_loader))

    for label, metric in metrics.items():
        metric.attach(trainer, label, "batch_wise")

    # define tqdm progress bar
    log_interval = 1
    desc = 'ITERATION - loss: {:.2f}'
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

    # define callbacks
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration_metrics = engine.state.metrics
        pbar.desc = desc.format(iteration_metrics['loss'])
        pbar.update(log_interval)

    wandb_logger = WandBLogger(
        project="homework1-cc7221",
        entity="p137",
        name="sperm-segmentation",
        config={"max_epochs": args.epochs, "batch_size": args.batch_size},
        tags=["pytorch-ignite", "sperm-seg"]
    )

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        metric_names="all",
        output_transform=lambda loss_: {"loss": loss_[2]},
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    wandb_logger.watch(model)
    trainer.run(train_loader, max_epochs=args.epochs)
    pbar.close()


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append('.')
    sys.path.append('..')

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data/SpermSegGS', type=str, help='Path to dataset folder')
    parser.add_argument('--val-size', default=0.05, type=float, help='Validation size')

    # training parameters
    parser.add_argument('--num-workers', default=0, type=int, help='Number of data loader workers.')
    parser.add_argument('--batch-size', default=2, type=int, help="Batch size")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')

    torch.cuda.empty_cache()
    args_ = parser.parse_args()
    run(args_)


