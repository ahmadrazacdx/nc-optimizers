"""CIFAR-10 data loading with standard augmentation."""

import os
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class FastCIFAR10(torchvision.datasets.CIFAR10):
    def _check_integrity(self):
        return True

def get_loaders(batch_size=128, data_dir='./data', num_workers=2):
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = FastCIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform)
    test_set = FastCIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def save_sample_grid(data_dir='./data', path='figures/sample_cifar10.png'):
    dataset = FastCIFAR10(
        root=data_dir, train=True, download=False, transform=T.ToTensor())
    classes = dataset.classes

    per_class = {c: [] for c in range(10)}
    for img, label in dataset:
        if len(per_class[label]) < 8:
            per_class[label].append(img)
        if all(len(v) == 8 for v in per_class.values()):
            break

    rows = []
    for c in range(10):
        rows.append(torch.stack(per_class[c]))

    fig, axes = plt.subplots(10, 8, figsize=(8, 10))
    for r in range(10):
        for col in range(8):
            ax = axes[r, col]
            ax.imshow(rows[r][col].permute(1, 2, 0).numpy())
            ax.set_axis_off()
        axes[r, 0].set_ylabel(classes[r], fontsize=8, rotation=0,
                               labelpad=40, va='center')

    fig.suptitle('CIFAR-10 samples (8 per class)', fontsize=13, y=0.92)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    train_loader, test_loader = get_loaders()
    print(f'Train: {len(train_loader.dataset):,} samples, '
          f'{len(train_loader)} batches of {train_loader.batch_size}')
    print(f'Test:  {len(test_loader.dataset):,} samples, '
          f'{len(test_loader)} batches of {test_loader.batch_size}')
    print(f'Classes ({len(train_loader.dataset.classes)}): '
          f'{", ".join(train_loader.dataset.classes)}')
    save_sample_grid()
