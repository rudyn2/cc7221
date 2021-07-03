import torch
from datasets import get_datasets
from termcolor import colored
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, ConfusionMatrix, DiceCoefficient, IoU, MetricsLambda, RunningAverage
from ignite.utils import to_onehot
from ignite.contrib.handlers import WandBLogger, global_step_from_engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from losses import FocalLoss, DiceLoss, WeightedPixelWiseNLLoss, cross_entropy_loss
from tqdm import tqdm
from unet_model import UNet
import torchvision


NUM_CLASSES = 4


class ReduceLROnPlateauScheduler:
    """Wrapper of torch.optim.lr_scheduler.ReduceLROnPlateau with __call__ method like other schedulers in
        contrib\handlers\param_scheduler.py"""

    def __init__(
            self,
            optimizer,
            metric_name,
            mode='min', factor=0.1, patience=10,
            threshold=1e-4, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-8, verbose=False,
    ):
        self.metric_name = metric_name
        self.scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                           threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                                           min_lr=min_lr, eps=eps, verbose=verbose)

    def __call__(self, engine, name=None):
        self.scheduler.step(engine.state.metrics[self.metric_name])

    def state_dict(self):
        return self.scheduler.state_dict()


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
    y_pred_one_hot = to_onehot(y_pred_, num_classes=NUM_CLASSES)
    return dict(y_pred=y_pred_one_hot, y=y_)  # output format is according to `DiceCoefficient` docs


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(colored("Using device: ", "white") + colored(device, "green"))

    print(colored("[*] Initializing dataset and dataloader", "white"))
    train, val, _ = get_datasets(args.data,
                                 use_validation=not args.all_train,
                                 mosaic_prob=args.mosaic_prob,
                                 new_size=args.new_size)
    print(colored("Total train examples: ", "white") + colored(len(train), "green"))
    print(colored("Total val examples: ", "white") + colored(len(val), "green"))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
    if not args.all_train:
        val_loader = DataLoader(val, batch_size=2, shuffle=True, num_workers=args.num_workers)
    print(colored("[+] Dataset & Dataloader Ready!", "green"))

    print(colored("[*] Initializing model, optimizer and loss", "white"))
    # model = DLv3Wrapper()
    model_factory = {
        'fcn-resnet50': lambda: torchvision.models.segmentation.fcn_resnet50(num_classes=NUM_CLASSES,
                                                                             pretrained=False),
        'fcn-resnet101': lambda: torchvision.models.segmentation.fcn_resnet101(num_classes=NUM_CLASSES,
                                                                               pretrained=False),
        'deeplab-resnet50': lambda: torchvision.models.segmentation.deeplabv3_resnet50(num_classes=NUM_CLASSES,
                                                                                       pretrained=False),
        'deeplab-resnet101': lambda: torchvision.models.segmentation.deeplabv3_resnet101(num_classes=NUM_CLASSES,
                                                                                         pretrained=False),
        'unet': lambda: UNet(n_channels=3, n_classes=4)
    }
    model = model_factory[args.model]()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateauScheduler(optimizer, metric_name="dice", mode="max", factor=0.5,
                                           patience=7, verbose=True)
    if args.loss == 'focal':
        loss = FocalLoss(apply_nonlin=torch.sigmoid)
    elif args.loss == 'wnll':
        loss = WeightedPixelWiseNLLoss(weights={
            0: args.weight0,
            1: args.weight1,
            2: args.weight2,
            3: args.weight3
        })
    elif args.loss == 'cross':
        loss = cross_entropy_loss()
    else:
        loss = DiceLoss()
    print(colored("[+] Model, optimizer and loss are ready!", "green"))

    print(colored("[*] Creating engine and handlers", "white"))
    score_function = lambda engine: engine.state.metrics['dice']
    avg_fn = lambda x: torch.mean(x).item()
    cm_metric = ConfusionMatrix(num_classes=NUM_CLASSES, output_transform=output_transform_seg)
    metrics = {'loss': Loss(loss_fn=loss, output_transform=lambda x: (x[0], x[1])),
               'loss_avg': RunningAverage(Loss(loss_fn=loss, output_transform=lambda x: (x[0], x[1]))),
               'dice': MetricsLambda(avg_fn, DiceCoefficient(cm_metric)),
               'iou': MetricsLambda(avg_fn, IoU(cm_metric)),
               'dice_sin_background': MetricsLambda(avg_fn, DiceCoefficient(cm_metric, ignore_index=0)),
               'iou_sin_background': MetricsLambda(avg_fn, IoU(cm_metric, ignore_index=0)),
               'dice_background': MetricsLambda(lambda x: x[0].item(), DiceCoefficient(cm_metric)),
               'dice_head': MetricsLambda(lambda x: x[1].item(), DiceCoefficient(cm_metric)),
               'dice_mid': MetricsLambda(lambda x: x[2].item(), DiceCoefficient(cm_metric)),
               'dice_tail': MetricsLambda(lambda x: x[3].item(), DiceCoefficient(cm_metric)),
               'iou_background': MetricsLambda(lambda x: x[0].item(), IoU(cm_metric)),
               'iou_head': MetricsLambda(lambda x: x[1].item(), IoU(cm_metric)),
               'iou_mid': MetricsLambda(lambda x: x[2].item(), IoU(cm_metric)),
               'iou_tail': MetricsLambda(lambda x: x[3].item(), IoU(cm_metric))
               }
    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        loss,
                                        prepare_batch=prepare_batch,
                                        output_transform=lambda x, y, y_, l: (y_, y, l),
                                        device=device)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, prepare_batch=prepare_batch)

    for label, metric in metrics.items():
        metric.attach(trainer, label, "batch_wise")

    wandb_logger = WandBLogger(
        project="homework1-cc7221",
        entity="p137",
        name="sperm-segmentation",
        config={"max_epochs": args.epochs,
                "patience": args.patience,
                "model": args.model,
                "lr": args.lr,
                "loss": args.loss,
                "mosaic_prob": args.mosaic_prob,
                "batch_size": args.batch_size},
        tags=["pytorch-ignite", "sperm-seg"]
    )
    model_checkpoint = ModelCheckpoint(
        wandb_logger.run.dir, n_saved=2, filename_prefix='best',
        require_empty=False, score_function=score_function,
        score_name="dice",
        global_step_transform=global_step_from_engine(trainer)
    )
    early_stopping_handler = EarlyStopping(patience=args.patience,
                                           score_function=score_function,
                                           min_delta=0.01,
                                           trainer=trainer)
    print(colored("[+] Engine and handlers are ready!", "green"))

    # define tqdm progress bar
    log_interval = 1
    desc = 'ITERATION [EPOCH={}]- loss: {:.2f}'
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0, 0))

    # define callbacks
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration_metrics = engine.state.metrics
        pbar.desc = desc.format(engine.state.epoch, iteration_metrics['loss'])
        pbar.update(log_interval)

    print(colored("[*] Attaching event handlers", "white"))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler=lambda _: train_evaluator.run(train_loader))

    # create evaluation engine and attach its handlers
    if not args.all_train:
        val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, prepare_batch=prepare_batch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler=lambda _: val_evaluator.run(val_loader))
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {'model': model})
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        wandb_logger.attach_output_handler(
            val_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names="all",
            global_step_transform=lambda *_: trainer.state.iteration,
        )
    else:
        train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)
        train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {'model': model})

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        metric_names="all",
        output_transform=lambda loss_: {"loss": loss_[2]},
        global_step_transform=lambda *_: trainer.state.iteration,
    )
    print(colored("[+] Event handlers are ready!", "green"))

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
    parser.add_argument('--data', default='../data/SpermSegGS', type=str, help='Path to dataset folder.')
    parser.add_argument('--new-size', default=None, type=str, help='Resize images before processing')
    parser.add_argument('--all-train', action='store_true', help='Whether to use all the data for training or not.')

    # training parameters
    parser.add_argument('--model', default='deeplab-resnet50', help='Model architecture.')
    parser.add_argument('--loss', default='dice', type=str, help='Type of loss function')
    parser.add_argument('--patience', default=7, type=int, help='Number of epochs without reduction in validation loss'
                                                                'until early stopping.')
    parser.add_argument('--num-workers', default=0, type=int, help='Number of data loader workers.')
    parser.add_argument('--batch-size', default=2, type=int, help="Batch size")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs")
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--mosaic-prob', default=0.5, type=float, help='Probability of applying mosaic augmentation')
    parser.add_argument('--weight0', default=0.1, type=float, help='weight for class 0')
    parser.add_argument('--weight1', default=0.30, type=float, help='weight for class 1')
    parser.add_argument('--weight2', default=0.30, type=float, help='weight for class 2')
    parser.add_argument('--weight3', default=0.30, type=float, help='weight for class 3')

    torch.cuda.empty_cache()
    args_ = parser.parse_args()
    if args_.new_size:
        args_.new_size = tuple([int(s) for s in str(args_.new_size).split(",")])
    run(args_)
