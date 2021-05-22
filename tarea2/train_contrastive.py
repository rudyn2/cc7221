import sys

sys.path.append('.')
sys.path.append('..')

from datasets import ContrastiveDataset, SketchTrainDataset, FlickrDataset
import argparse
import wandb
from models import ResNet34, SiameseNetwork
from losses import contrastive_loss

if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    import torch

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flickr', default='data/Flickr25K', type=str, help='Path to flickr dataset folder')
    parser.add_argument('--sketches', default='data/Sketch_EITZ', type=str,
                        help='Path to sketches dataset folder')
    parser.add_argument('--sketches-backbone-weights', default='../weights/sketches.pth',
                        help='Path to sketches weights')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--val-size', default=0.1, type=float,
                        help='Ratio of train dataset that will be used for validation')
    parser.add_argument('--n-similar', default=3, type=int, help='Number of similar flickr-sketch pairs to be created'
                                                                 'per sketch')
    parser.add_argument('--m-different', default=3, type=int,
                        help='Number of different flickr-sketch pairs to be created'
                             'per sketch')
    parser.add_argument('--optimizer', default='SGD', type=str, help='Type of optimizer [adam, sgd]')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD Optimizer momentum')
    parser.add_argument('--t-0', default=8000, type=int, help='Cosine Decay number of iterations to restart.')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = args.device
    train_flickr = FlickrDataset(args.flickr)
    train_sketches = SketchTrainDataset(args.sketches)
    dataset = ContrastiveDataset(flickr_dataset=train_flickr, sketches_dataset=train_sketches,
                                 n_similar=args.n_similar, m_different=args.m_different)

    n_val = int(len(dataset) * args.val_size)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batch_size)
    val_loader = DataLoader(val, batch_size=args.batch_size // 4)

    # load backbones
    print("[*] Initializing weights...")
    imagenet_net = ResNet34()
    sketches_net = ResNet34()
    sketches_net.load_state_dict(torch.load(args.sketches_backbone_weights))
    print("[+] Weights loaded")

    print("[*] Adapting output layers...")
    sketches_net.adapt_fc()
    imagenet_net.adapt_fc()
    print("[+] Layers successfully adapted")

    print("[*] Initializing model, loss and optimizer")
    contrastive_net = SiameseNetwork(sketches_net, imagenet_net)
    contrastive_net.to(args.device)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(contrastive_net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(contrastive_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0)
    contrastive_loss = contrastive_loss()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    print("[+] Model, loss and optimizer were initialized successfully")

    if not args.debug:
        wandb.init(project='homework1-cc7221', entity='p137')
        config = wandb.config
        config.model = contrastive_net.__class__.__name__ + '_contrastive'
        config.device = device
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.learning_rate = args.lr
        config.optimizer = optimizer.__class__.__name__
        config.train_size = n_train
        config.val_size = n_val

    print("[*] Training")
    best_avg_acc = 0
    for epoch in range(args.epochs):

        # Train
        tag = ''  # tag = '*' if the model was saved in the last epoch
        train_total_loss = 0
        running_acc_flickr, running_acc_sketches = 0, 0
        for i, (first, second, first_label, second_label, similarity) in enumerate(train_loader):
            # first: flickr, second: sketches
            first, second, similarity = first.to(args.device), second.to(args.device), similarity.to(args.device)
            first_label, second_label = first_label.to(args.device), second_label.to(args.device)
            pred = contrastive_net.forward((first, second, None), include_negative=False)
            pred_feats_first = pred['feats']['anchor']
            pred_feats_second = pred['feats']['positive']
            pred_logits_first = pred['logits']['anchor']
            pred_logits_second = pred['logits']['positive']

            # region: optimization step
            optimizer.zero_grad()
            loss_contrastive = contrastive_loss(pred_feats_first, pred_feats_second, similarity)
            loss_classifier_sketches = cross_entropy_loss(pred_logits_first, first_label)
            loss_classifier_flickr = cross_entropy_loss(pred_logits_second, second_label)
            loss = 0.6 * (0.5 * (loss_classifier_flickr + loss_classifier_sketches)) + 0.4 * loss_contrastive
            loss.backward()
            optimizer.step()
            # endregion

            train_total_loss += loss.item()
            avg_train_loss = train_total_loss / (i + 1)

            items = min(n_train, (i + 1) * train_loader.batch_size)

            # accuracy flickr
            _, max_idx = torch.max(pred_logits_first, dim=1)
            running_acc_flickr += torch.sum(max_idx == first_label).item()
            avg_flickr_train_acc = running_acc_flickr / items * 100

            # accuracy sketches
            _, max_idx = torch.max(pred_logits_second, dim=1)
            running_acc_sketches += torch.sum(max_idx == second_label).item()
            avg_sketches_train_acc = running_acc_sketches / items * 100

            scheduler.step()
            sys.stdout.write('\r')
            sys.stdout.write(f"Epoch: {epoch + 1}({i}/{len(train_loader)})| "
                             f"Train[Loss: {avg_train_loss:.5f}, "
                             f"flickr acc: {avg_flickr_train_acc:.3f}%, "
                             f"sketches acc: {avg_sketches_train_acc:.3f}%")
            if not args.debug:
                wandb.log({'train/loss': avg_train_loss, 'train/acc flickr': avg_flickr_train_acc,
                           'train/acc sketches': avg_sketches_train_acc})

        train_loader.dataset.dataset.on_epoch_end()
        avg_train_loss = train_total_loss / len(train_loader)

        if not args.debug:
            wandb.log({'train/loss': avg_train_loss, 'epoch': epoch + 1})

        # Validate
        val_total_loss = 0
        best_val_loss = 1e100
        running_val_acc = 0
        running_acc_flickr, running_acc_sketches = 0, 0
        avg_val_flicker_acc, avg_sketches_val_acc = 0, 0
        for i, (first, second, first_label, second_label, similarity) in enumerate(val_loader):
            # first: flickr, second: sketches
            first, second, similarity = first.to(args.device), second.to(args.device), similarity.to(args.device)
            first_label, second_label = first_label.to(args.device), second_label.to(args.device)

            pred = contrastive_net.forward((first, second, None), include_negative=False)

            pred_feats_first = pred['feats']['anchor']
            pred_feats_second = pred['feats']['positive']
            pred_logits_first = pred['logits']['anchor']
            pred_logits_second = pred['logits']['positive']

            loss_contrastive = contrastive_loss(pred_feats_first, pred_feats_second, similarity)
            loss_classifier_sketches = cross_entropy_loss(pred_logits_first, first_label)
            loss_classifier_flickr = cross_entropy_loss(pred_logits_second, second_label)
            loss = 0.6 * (0.5 * (loss_classifier_flickr + loss_classifier_sketches)) + 0.4 * loss_contrastive

            val_total_loss += loss.item()
            avg_val_loss = val_total_loss / (i + 1)

            items = min(n_train, (i + 1) * train_loader.batch_size)

            # accuracy flickr
            _, max_idx = torch.max(pred_logits_first, dim=1)
            running_acc_flickr += torch.sum(max_idx == first_label).item()
            avg_val_flicker_acc = running_acc_flickr / items * 100

            # accuracy sketches
            _, max_idx = torch.max(pred_logits_second, dim=1)
            running_acc_sketches += torch.sum(max_idx == second_label).item()
            avg_sketches_val_acc = running_acc_sketches / items * 100

        avg_val_loss = val_total_loss / len(val_loader)

        if not args.debug:
            wandb.log({'val/loss': avg_val_loss, 'val/acc flickr': avg_val_flicker_acc,
                       'val/acc sketches': avg_sketches_val_acc, 'epoch': epoch + 1})

            # checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            model_name = f"best_{contrastive_net.__class__.__name__}_contrastive.pth"
            torch.save(contrastive_net.state_dict(), model_name)
            if not args.debug:
                wandb.save(model_name)
            tag = '*'

            sys.stdout.write(
                f", Val[Loss: {avg_val_loss}, flickr acc: {avg_val_flicker_acc}, sketches acc: {avg_sketches_val_acc} {tag}")
            sys.stdout.flush()
            sys.stdout.write('\n')
