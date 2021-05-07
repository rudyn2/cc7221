import sys

sys.path.append('.')
sys.path.append('..')

from tarea2.dataset import SketchTrainDataset
import argparse
import wandb
from model_sketchz import ResNet34

if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    import torch

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../dataset/Sketch_EITZ', type=str, help='Path to embeddings hdf5')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--val-size', default=0.1, type=float,
                        help='Ratio of train dataset that will be used for validation')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = args.device
    dataset = SketchTrainDataset(args.data)

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=args.batch_size)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    mse_loss = torch.nn.CrossEntropyLoss()

    model = ResNet34()
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.debug:
        wandb.init(project='tsad', entity='autonomous-driving')
        config = wandb.config
        config.model = model.__class__.__name__
        config.device = device
        config.batch_size = args.batch_size
        config.hidden_size = args.hidden_size
        config.epochs = args.epochs
        config.learning_rate = args.lr

    best_avg_acc = 0
    for epoch in range(args.epochs):

        # Train
        tag = ''  # tag = '*' if the model was saved in the last epoch
        train_total_loss = 0
        running_acc = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            pred = model(images)

            # optimization step
            optimizer.zero_grad()
            loss = mse_loss(pred, labels)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

            avg_train_loss = train_total_loss / (i + 1)

            # accuracy
            items = min(n_train, (i + 1) * train_loader.batch_size)
            _, max_idx = torch.max(pred, dim=1)
            running_acc += torch.sum(max_idx == labels).item()
            avg_acc = running_acc / items * 100

            sys.stdout.write('\r')
            sys.stdout.write(f"Epoch: {epoch + 1}({i}/{len(train_loader)})| "
                             f"Train loss: {avg_train_loss:.5f}| "
                             f"Train acc: {avg_acc:.3f}%")
            if not args.debug:
                wandb.log({'train/loss': avg_train_loss})

        avg_train_loss = train_total_loss / len(train_loader)

        if not args.debug:
            wandb.log({'train/loss': avg_train_loss, 'epoch': epoch + 1})

        # Validate
        val_total_loss = 0
        running_val_acc = 0
        for i, (images, labels) in enumerate(val_loader):
            with torch.no_grad():
                images, labels = images.to(args.device), labels.to(args.device)
                pred = model(images)
                loss = mse_loss(pred, labels)
                val_total_loss += loss.item()

                _, max_idx = torch.max(pred, dim=1)
                running_val_acc += torch.sum(max_idx == labels).item()

        avg_val_acc = running_val_acc / n_val * 100
        avg_val_loss = val_total_loss / len(val_loader)

        if not args.debug:
            wandb.log({'val/loss': avg_val_loss, 'epoch': epoch + 1})

        # checkpointing
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            model_name = f"best_{model.__class__.__name__}.pth"
            torch.save(model.state_dict(), model_name)
            if not args.debug:
                wandb.save(model_name)
            tag = '*'

        sys.stdout.write(f", Val loss: {avg_val_loss} | Val acc: {avg_val_acc} {tag}")
        sys.stdout.flush()
        sys.stdout.write('\n')
