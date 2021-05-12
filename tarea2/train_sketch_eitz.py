import sys

sys.path.append('.')
sys.path.append('..')

from tarea2.datasets import ContrastiveDataset, SketchTrainDataset, FlickrDataset, TripletDataset
import argparse
import wandb
from models import ResNet34, SiameseNetwork
from losses import contrastive_loss, triplet_loss

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
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--val-size', default=0.1, type=float,
                        help='Ratio of train dataset that will be used for validation')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print("[*] Initializing data")
    device = args.device
    train_flickr = FlickrDataset(args.flickr)
    train_sketches = SketchTrainDataset(args.sketches)
    dataset = TripletDataset(flickr_dataset=train_flickr, sketches_dataset=train_sketches)

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batch_size)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    print("[+] Dataset initialized successfully")

    # load backbones
    print("[*] Initializing weights...")
    imagenet_net = ResNet34()
    sketches_net = ResNet34()
    sketches_net.load_state_dict(torch.load('/home/rudy/Documents/cc7221/tarea2/weights/sketches.pth'))
    print("[+] Weights loaded")

    print("[*] Adapting output layers...")
    sketches_net.adapt_fc()
    imagenet_net.adapt_fc()
    print("[+] Layers successfully adapted")

    print("[*] Initializing model, loss and optimizer")
    siamese_net = SiameseNetwork(sketches_net, imagenet_net)
    siamese_net.to(args.device)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=args.lr)
    triplet_loss = triplet_loss()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    print("[+] Model, loss and optimizer were initialized successfully")

    if not args.debug:
        wandb.init(project='homework1-cc7221', entity='p137')
        config = wandb.config
        config.model = siamese_net.__class__.__name__ + "_triplet"
        config.device = device
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.learning_rate = args.lr

    print("[*] Training")
    best_avg_acc = 0
    for epoch in range(args.epochs):

        # Train
        tag = ''  # tag = '*' if the model was saved in the last epoch
        train_total_loss = 0
        running_acc_anchor, running_acc_positive, running_acc_negative = 0, 0, 0
        for i, (a, p, n, a_l, p_l, n_l) in enumerate(train_loader):
            # first: flickr, second: sketches
            a, p, n = a.to(args.device), p.to(args.device), n.to(args.device)
            a_l, p_l, n_l = a_l.to(args.device), p_l.to(args.device), n_l.to(args.device)
            pred = siamese_net.forward((a, p, n), include_negative=True)

            pred_feats_anchor = pred['feats']['anchor']
            pred_feats_positive = pred['feats']['positive']
            pred_feats_negative = pred['feats']['negative']

            pred_logits_anchor = pred['logits']['anchor']
            pred_logits_positive = pred['logits']['positive']
            pred_logits_negative = pred['logits']['negative']

            # region: optimization step
            optimizer.zero_grad()
            loss_triplet = triplet_loss(pred_feats_anchor, pred_feats_positive, pred_feats_negative)
            loss_classifier_anchor = cross_entropy_loss(pred_logits_anchor, a_l)
            loss_classifier_positive = cross_entropy_loss(pred_logits_positive, p_l)
            loss_classifier_negative = cross_entropy_loss(pred_logits_negative, n_l)
            loss = 0.6 * ((1/3)* (loss_classifier_anchor + loss_classifier_positive + loss_classifier_negative)) + 0.4 * loss_triplet
            loss.backward()
            optimizer.step()
            # endregion

            train_total_loss += loss.item()
            avg_train_loss = train_total_loss / (i + 1)

            items = min(n_train, (i + 1) * train_loader.batch_size)

            # accuracy anchor
            _, max_idx = torch.max(pred_logits_anchor, dim=1)
            running_acc_anchor += torch.sum(max_idx == a_l).item()
            avg_acc_train_anchor = running_acc_anchor / items * 100

            # accuracy positive
            _, max_idx = torch.max(pred_feats_positive, dim=1)
            running_acc_positive += torch.sum(max_idx == p_l).item()
            avg_acc_train_positive = running_acc_positive / items * 100

            # accuracy negative
            _, max_idx = torch.max(pred_feats_negative, dim=1)
            running_acc_negative += torch.sum(max_idx == n_l).item()
            avg_acc_train_negative = running_acc_negative / items * 100

            sys.stdout.write('\r')
            sys.stdout.write(f"Epoch: {epoch + 1}({i}/{len(train_loader)})| "
                             f"Train[Loss: {avg_train_loss:.5f}, "
                             f"anchor acc: {avg_acc_train_anchor:.3f}%, "
                             f"positive acc: {avg_acc_train_positive:.3f}%, "
                             f"negative acc: {avg_acc_train_negative:.3f}%")
            if not args.debug:
                wandb.log({'train/loss': avg_train_loss,
                           'train/acc anchor': avg_acc_train_anchor,
                           'train/acc positive': avg_acc_train_positive,
                           'train/acc negative': avg_acc_train_negative})

        avg_train_loss = train_total_loss / len(train_loader)

        if not args.debug:
            wandb.log({'train/loss': avg_train_loss, 'epoch': epoch + 1})

        # Validate
        val_total_loss = 0
        best_val_loss = 1e100
        running_val_acc = 0
        running_acc_anchor, running_acc_positive, running_acc_negative = 0, 0, 0
        avg_acc_val_anchor, avg_acc_val_positive, avg_acc_val_negative = 0, 0, 0
        for i, (a, p, n, a_l, p_l, n_l) in enumerate(val_loader):
            a, p, n = a.to(args.device), p.to(args.device), n.to(args.device)
            a_l, p_l, n_l = a_l.to(args.device), p_l.to(args.device), n_l.to(args.device)
            pred = siamese_net.forward((a, p, n), include_negative=True)

            pred_feats_anchor = pred['feats']['anchor']
            pred_feats_positive = pred['feats']['positive']
            pred_feats_negative = pred['feats']['negative']

            pred_logits_anchor = pred['logits']['anchor']
            pred_logits_positive = pred['logits']['positive']
            pred_logits_negative = pred['logits']['negative']

            loss_triplet = triplet_loss(pred_feats_anchor, pred_feats_positive, pred_feats_negative)
            loss_classifier_anchor = cross_entropy_loss(pred_logits_anchor, a_l)
            loss_classifier_positive = cross_entropy_loss(pred_logits_positive, p_l)
            loss_classifier_negative = cross_entropy_loss(pred_logits_negative, n_l)
            loss = 0.6 * ((1/3) * (loss_classifier_anchor + loss_classifier_positive + loss_classifier_negative)) + 0.4 * loss_triplet

            val_total_loss += loss.item()
            avg_total_loss = val_total_loss / (i + 1)

            items = min(n_train, (i + 1) * train_loader.batch_size)

            # accuracy anchor
            _, max_idx = torch.max(pred_logits_anchor, dim=1)
            running_acc_anchor += torch.sum(max_idx == a_l).item()
            avg_acc_val_anchor = running_acc_anchor / items * 100

            # accuracy positive
            _, max_idx = torch.max(pred_feats_positive, dim=1)
            running_acc_positive += torch.sum(max_idx == p_l).item()
            avg_acc_val_positive = running_acc_positive / items * 100

            # accuracy negative
            _, max_idx = torch.max(pred_feats_negative, dim=1)
            running_acc_negative += torch.sum(max_idx == n_l).item()
            avg_acc_val_negative = running_acc_negative / items * 100

        avg_val_loss = val_total_loss

        if not args.debug:
            wandb.log({'val/loss': avg_val_loss,
                       'val/acc anchor': avg_acc_val_anchor,
                       'val/acc positive': avg_acc_val_positive,
                       'val/acc negative': avg_acc_val_negative,
                       'epoch': epoch + 1})

            # checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            model_name = f"best_{siamese_net.__class__.__name__}_triplet.pth"
            torch.save(siamese_net.state_dict(), model_name)
            if not args.debug:
                wandb.save(model_name)
            tag = '*'

            sys.stdout.write(
                f", Val[Loss: {avg_val_loss}, "
                f"anchor acc: {avg_acc_val_anchor}, "
                f"positive acc: {avg_acc_val_positive} "
                f"negative acc: {avg_acc_val_negative} {tag}")
            sys.stdout.flush()
            sys.stdout.write('\n')
