"""Train ResNet-18 on CIFAR-10 with different optimizers."""

import os
import argparse
import torch
import torch.nn as nn
from data import get_loaders
from model import ResNet18
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)
        
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)
        
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw'], default='sgd')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=None, help='Default depends on optimizer')
    parser.add_argument('--weight_decay', type=float, default=None, help='Default depends on optimizer')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.optimizer == 'sgd':
        lr = args.lr if args.lr is not None else 0.1
        wd = args.weight_decay if args.weight_decay is not None else 5e-4
    elif args.optimizer == 'adam':
        lr = args.lr if args.lr is not None else 1e-3
        wd = args.weight_decay if args.weight_decay is not None else 5e-4
    elif args.optimizer == 'adamw':
        lr = args.lr if args.lr is not None else 1e-3
        wd = args.weight_decay if args.weight_decay is not None else 5e-2
        
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    os.makedirs('checkpoints', exist_ok=True)

    print(f"--- Training with {args.optimizer.upper()} ---")
    print(f"Device: {device}")
    print(f"LR: {lr}, Weight Decay: {wd}, Epochs: {args.epochs}, Batch: {args.batch_size}")
    
    train_loader, test_loader = get_loaders(batch_size=args.batch_size)
    
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 65)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == args.epochs:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc*100:8.2f}% | {test_loss:9.4f} | {test_acc*100:7.2f}%")
            
            ckpt_path = f"checkpoints/{args.optimizer}_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, ckpt_path)
    
    final_path = f"checkpoints/{args.optimizer}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    main()
