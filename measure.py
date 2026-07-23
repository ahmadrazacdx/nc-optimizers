"""Compute Neural Collapse metrics."""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet18
from data import get_loaders


def extract_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.features(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_features), torch.cat(all_labels)


def compute_metrics(features, labels, W, b):
    C = W.shape[0]
    d = features.shape[1]
    global_mean = features.mean(dim=0)
    class_means = torch.zeros(C, d)
    for c in range(C):
        class_means[c] = features[labels == c].mean(dim=0)
        
    # Within-class variability collapse
    Sigma_W = torch.zeros(d, d)
    for c in range(C):
        class_features = features[labels == c]
        centered = class_features - class_means[c]
        Sigma_W += (centered.T @ centered) / len(class_features)
    Sigma_W /= C
    
    centered_means = class_means - global_mean
    Sigma_B = (centered_means.T @ centered_means) / C
    pinv_B = torch.linalg.pinv(Sigma_B)
    nc1 = torch.trace(Sigma_W @ pinv_B).item() / C
    
    # Convergence to Simplex ETF
    M_hat = centered_means / torch.norm(centered_means, dim=1, keepdim=True)
    ETF = torch.eye(C) - (1.0 / C) * torch.ones(C, C)
    ETF *= (C / (C - 1.0))
    nc2 = torch.norm(M_hat @ M_hat.T - ETF, p='fro').item()
    
    # Classifier-feature self-duality
    centered_W = W - W.mean(dim=0)
    W_hat = centered_W / torch.norm(centered_W, dim=1, keepdim=True)
    nc3 = torch.norm(W_hat - M_hat, p='fro').item()
    
    # NCC agreement
    logits = features @ W.T + b
    model_preds = logits.argmax(dim=1)
    ncc_logits = features @ class_means.T - 0.5 * torch.norm(class_means, dim=1)**2
    ncc_preds = ncc_logits.argmax(dim=1)
    
    nc4 = 1.0 - (model_preds == ncc_preds).float().mean().item()
    
    return nc1, nc2, nc3, nc4, M_hat


def plot_metrics(all_results, optimizers, epochs):
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = [
        'Within-class collapse',
        'Simplex ETF', 
        'Self-duality', 
        'NCC agreement'
    ]
    
    colors = {'sgd': '#D55E00', 'adam': '#0072B2', 'adamw': '#009E73'}
    
    for i, ax in enumerate(axes.flatten()):
        for opt in optimizers:
            if opt in all_results:
                ax.plot(epochs, all_results[opt][:, i], marker='o', 
                        label=opt.upper(), color=colors[opt], linewidth=2)
        
        ax.set_title(metrics[i], fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        
        if i == 0:
            ax.set_yscale('log')
            ax.legend(frameon=False)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.5)
            
    plt.tight_layout()
    plt.savefig('figures/nc_metrics.png', dpi=150, bbox_inches='tight')
    print("Saved figures/nc_metrics.png")


def plot_training_curves(all_accs, optimizers, epochs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {'sgd': '#D55E00', 'adam': '#0072B2', 'adamw': '#009E73'}
    
    for opt in optimizers:
        if opt in all_accs:
            train_accs = [x * 100 for x in all_accs[opt]['train']]
            test_accs = [x * 100 for x in all_accs[opt]['test']]
            axes[0].plot(epochs, train_accs, marker='o', label=opt.upper(), color=colors[opt], linewidth=2)
            axes[1].plot(epochs, test_accs, marker='o', label=opt.upper(), color=colors[opt], linewidth=2)
            
    axes[0].set_title('Train Accuracy (%)', fontweight='bold')
    axes[1].set_title('Test Accuracy (%)', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(frameon=False)
        
    plt.tight_layout()
    plt.savefig('figures/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/training_curves.png")


def plot_cosine_similarity(M_hat, opt_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    cos_sim = (M_hat @ M_hat.T).numpy()
    
    im = ax.imshow(cos_sim, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    
    ax.set_title(f'ETF Cosine Similarity ({opt_name.upper()})', fontweight='bold')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    
    for i in range(10):
        for j in range(10):
            val = cos_sim[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)
            
    plt.tight_layout()
    plt.savefig(f'figures/cosine_similarity_{opt_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figures/cosine_similarity_{opt_name}.png")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_loader, _ = get_loaders(batch_size=256, num_workers=0)
    
    optimizers = ['sgd', 'adam', 'adamw']
    epochs = list(range(10, 351, 10))
    all_results = {}
    all_accs = {}
    
    for opt in optimizers:
        if not os.path.exists(f'checkpoints/{opt}_final.pt'):
            print(f"Skipping {opt}, final checkpoint not found.")
            continue
        opt_metrics = []
        train_accs, test_accs = [], []
        last_M_hat = None
        
        for ep in epochs:
            ckpt_path = f'checkpoints/{opt}_epoch{ep}.pt'
            if not os.path.exists(ckpt_path):
                print(f"  Missing {ckpt_path}, stopping at epoch {ep-10}.")
                break
                
            model = ResNet18(num_classes=10).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            
            train_accs.append(ckpt['train_acc'])
            test_accs.append(ckpt['test_acc'])
            
            features, labels = extract_features(model, train_loader, device)
            
            W = model.head.weight.detach().cpu()
            b = model.head.bias.detach().cpu()
            
            nc1, nc2, nc3, nc4, M_hat = compute_metrics(features, labels, W, b)
            opt_metrics.append([nc1, nc2, nc3, nc4])
            last_M_hat = M_hat
            print(f"  Ep {ep:3d} | NC1: {nc1:7.4f} | NC2: {nc2:6.4f} | NC3: {nc3:6.4f} | NC4: {nc4:6.4f}")
            
        all_results[opt] = np.array(opt_metrics)
        all_accs[opt] = {'train': train_accs, 'test': test_accs}
        
        if last_M_hat is not None:
            plot_cosine_similarity(last_M_hat, opt)
        
    if all_results:
        first_opt = list(all_results.keys())[0]
        actual_epochs = epochs[:len(all_results[first_opt])]
        plot_metrics(all_results, optimizers, actual_epochs)
        plot_training_curves(all_accs, optimizers, actual_epochs)
        
        export_data = {
            'epochs': actual_epochs,
            'optimizers': list(all_results.keys()),
            'metrics': {opt: all_results[opt].tolist() for opt in all_results},
            'accuracies': all_accs
        }
        with open('figures/metrics_data.json', 'w') as f:
            json.dump(export_data, f, indent=4)
        print("Saved raw data to figures/metrics_data.json")
    else:
        print("\nNo checkpoints found.")


if __name__ == '__main__':
    main()
