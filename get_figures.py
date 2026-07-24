import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(all_results, optimizers, epochs):
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = [
        'Within-class collapse (NC1)',
        'Simplex ETF (NC2)', 
        'Self-duality (NC3)', 
        'NCC agreement (NC4)'
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
    os.makedirs('figures', exist_ok=True)
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
    os.makedirs('figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cos_sim = M_hat @ M_hat.T
    
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
    json_path = 'artifacts/metrics_data.json'
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run measure.py first.")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    epochs = data['epochs']
    optimizers = data['optimizers']
    all_accs = data['accuracies']
    all_results = {opt: np.array(metrics) for opt, metrics in data['metrics'].items()}
    plot_metrics(all_results, optimizers, epochs)
    plot_training_curves(all_accs, optimizers, epochs)
    if 'm_hats' in data:
        for opt, m_hat_list in data['m_hats'].items():
            M_hat = np.array(m_hat_list)
            plot_cosine_similarity(M_hat, opt)


if __name__ == '__main__':
    main()
