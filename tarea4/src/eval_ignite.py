import torch
from torch.utils.data import DataLoader
from datasets import get_datasets
from termcolor import colored
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import ConfusionMatrix, DiceCoefficient, IoU, MetricsLambda
from ignite.utils import to_onehot
import torchvision


NUM_CLASSES = 4


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

    print(colored("Initializing test dataset...", color="white"))
    _, _, test_dataset = get_datasets(args.data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    model_factory = {
        'fcn-resnet50': lambda: torchvision.models.segmentation.fcn_resnet50(num_classes=NUM_CLASSES,
                                                                             pretrained=False),
        'fcn-resnet101': lambda: torchvision.models.segmentation.fcn_resnet101(num_classes=NUM_CLASSES,
                                                                               pretrained=False),
        'deeplab-resnet50': lambda: torchvision.models.segmentation.deeplabv3_resnet50(num_classes=NUM_CLASSES,
                                                                                       pretrained=False),
        'deeplab-resnet101': lambda: torchvision.models.segmentation.deeplabv3_resnet101(num_classes=NUM_CLASSES,
                                                                                         pretrained=False)
    }
    model = model_factory[args.model]()
    model.load_state_dict(torch.load(args.weights))
    model.to(device)

    cm_metric = ConfusionMatrix(num_classes=NUM_CLASSES, output_transform=output_transform_seg)
    metrics = {'dice': MetricsLambda(lambda x: torch.mean(x).item(), DiceCoefficient(cm_metric)),
               'iou': MetricsLambda(lambda x: torch.mean(x).item(), IoU(cm_metric)),
               'dice_background': MetricsLambda(lambda x: x[0].item(), DiceCoefficient(cm_metric)),
               'dice_head': MetricsLambda(lambda x: x[1].item(), DiceCoefficient(cm_metric)),
               'dice_mid': MetricsLambda(lambda x: x[2].item(), DiceCoefficient(cm_metric)),
               'dice_tail': MetricsLambda(lambda x: x[3].item(), DiceCoefficient(cm_metric)),
               'iou_background': MetricsLambda(lambda x: x[0].item(), IoU(cm_metric)),
               'iou_head': MetricsLambda(lambda x: x[1].item(), IoU(cm_metric)),
               'iou_mid': MetricsLambda(lambda x: x[2].item(), IoU(cm_metric)),
               'iou_tail': MetricsLambda(lambda x: x[3].item(), IoU(cm_metric))
               }

    print(colored("Evaluating...\n", color="white"))
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, prepare_batch=prepare_batch)

    @test_evaluator.on(Events.COMPLETED)
    def log_training_loss(engine):
        for k, v in engine.state.metrics.items():
            print(f"{k}: {v:.4f}")

    test_evaluator.run(test_loader)


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append('.')
    sys.path.append('..')

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data/SpermSegGS', type=str, help='Path to dataset folder.')
    parser.add_argument('--model', default='deeplab-resnet50', help='Model architecture.')
    parser.add_argument('--weights', required=True, help='Path to model weights.')
    parser.add_argument('--batch-size', default=2, type=int, help="Batch size")

    torch.cuda.empty_cache()
    args_ = parser.parse_args()
    run(args_)