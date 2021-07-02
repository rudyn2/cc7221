import matplotlib.pyplot as plt
import torch
from datasets import get_datasets
from termcolor import colored
import torchvision
from torch.utils.data import DataLoader


NUM_CLASSES = 4


def prepare_batch(batch, device, non_blocking):
    x = batch[0].to(device, non_blocking=non_blocking)
    y = batch[1].to(device, non_blocking=non_blocking)
    y = y.squeeze(1)
    return x, y


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test = get_datasets(args.data)
    test_loader = DataLoader(test, batch_size=2, num_workers=0)
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
    # model.load_state_dict(torch.load(args.weights))
    model.to(device)

    print(colored("Evaluating", "white"))
    images, gt_seg, pd_seg = [], [], []
    for i, batch in enumerate(test_loader):
        image, seg = prepare_batch(batch, device=device, non_blocking=True)
        image = image.to(device)
        predicted_seg = model(image)
        images.append(image.detach().cpu())
        gt_seg.append(seg.detach().cpu())
        pd_seg.append(predicted_seg['out'].detach().cpu())
    images = torch.cat(images, dim=0)
    gt_seg = torch.cat(gt_seg, dim=0)
    pd_seg = torch.cat(pd_seg, dim=0)

    print(colored("Plotting", "white"))
    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(14, 14))
    axs[0][0].set_title("Image")
    axs[0][1].set_title("Expected segmentation")
    axs[0][2].set_title("Predicted segmentation")
    for i in range(images.shape[0]):
        # plot image
        axs[i][0].imshow(images[i].permute(1, 2, 0).cpu())
        axs[i][0].axis('off')

        # plot gt seg
        axs[i][1].imshow(gt_seg[i].permute(1, 2, 0).cpu(), cmap='gray')
        axs[i][1].axis('off')

        # plot pd seg
        axs[i][2].imshow(pd_seg[i].permute(1, 2, 0).cpu(), cmap='gray')
        axs[i][2].axis('off')

    plt.tight_layout()
    plt.savefig("output.svg")
    plt.show()


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append('.')
    sys.path.append('..')

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data/SpermSegGS', type=str, help='Path to dataset folder.')
    parser.add_argument('--model', default='deeplab-resnet50', help='Model architecture.')
    parser.add_argument('--weights', required=False, help='Path to model weights.')
    parser.add_argument('--batch-size', default=2, type=int, help="Batch size")

    torch.cuda.empty_cache()
    args_ = parser.parse_args()
    run(args_)