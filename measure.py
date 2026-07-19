"""Compute Neural Collapse metrics."""

import os
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
    
    return nc1, nc2, nc3, nc4


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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_loader, _ = get_loaders(batch_size=256, num_workers=0)
    
    optimizers = ['sgd', 'adam', 'adamw']
    epochs = list(range(10, 201, 10))
    all_results = {}
    
    for opt in optimizers:
        if not os.path.exists(f'checkpoints/{opt}_final.pt'):
            print(f"Skipping {opt}, final checkpoint not found.")
            continue
        opt_metrics = []
        for ep in epochs:
            ckpt_path = f'checkpoints/{opt}_epoch{ep}.pt'
            if not os.path.exists(ckpt_path):
                print(f"  Missing {ckpt_path}, stopping at epoch {ep-10}.")
                break
                
            model = ResNet18(num_classes=10).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            
            features, labels = extract_features(model, train_loader, device)
            
            W = model.head.weight.detach().cpu()
            b = model.head.bias.detach().cpu()
            
            nc1, nc2, nc3, nc4 = compute_metrics(features, labels, W, b)
            opt_metrics.append([nc1, nc2, nc3, nc4])
            print(f"Ep {ep:3d} | NC1: {nc1:7.4f} | NC2: {nc2:6.4f} | NC3: {nc3:6.4f} | NC4: {nc4:6.4f}")
            
        all_results[opt] = np.array(opt_metrics)
        
    if all_results:
        first_opt = list(all_results.keys())[0]
        actual_epochs = epochs[:len(all_results[first_opt])]
        plot_metrics(all_results, optimizers, actual_epochs)
    else:
        print("\nNo checkpoints found.")


if __name__ == '__main__':
    main()
