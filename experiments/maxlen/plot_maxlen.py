import matplotlib.pyplot as plt
import numpy as np

def parse_log(filepath, start_line=69):
    epochs, train_loss, val_loss, auc, f1 = [], [], [], [], []
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            if i < start_line:
                continue
            parts = line.strip().split()
            if len(parts) == 5:
                epochs.append(int(parts[0]))
                train_loss.append(float(parts[1]))
                val_loss.append(float(parts[2]))
                auc.append(float(parts[3]))
                f1.append(float(parts[4]))
    return np.array(epochs), np.array(train_loss), np.array(val_loss), np.array(auc), np.array(f1)

maxlens = [30, 50, 100, 300, 512]
data = {}
for ml in maxlens:
    data[ml] = parse_log(f"experiments/maxlen/{ml}.out")

colors = {30: '#e41a1c', 50: '#377eb8', 100: '#4daf4a', 300: '#984ea3', 512: '#ff7f00'}

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("DeBERTa-v3-xsmall AV: Max Length Comparison (BS=100, LR=2e-5, 200 epochs)", fontsize=13, fontweight="bold")

# Training Loss
ax = axes[0, 0]
for ml in maxlens:
    e, tl, vl, auc, f1 = data[ml]
    ax.plot(e, tl, label=f"MaxLen={ml}", color=colors[ml], alpha=0.8)
ax.set_title("Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Validation Loss
ax = axes[0, 1]
for ml in maxlens:
    e, tl, vl, auc, f1 = data[ml]
    ax.plot(e, vl, label=f"MaxLen={ml}", color=colors[ml], alpha=0.8)
ax.set_title("Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Dev AUC
ax = axes[1, 0]
for ml in maxlens:
    e, tl, vl, auc, f1 = data[ml]
    ax.plot(e, auc, label=f"MaxLen={ml}", color=colors[ml], alpha=0.8)
ax.set_title("Dev AUC")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Dev F1 Score
ax = axes[1, 1]
for ml in maxlens:
    e, tl, vl, auc, f1 = data[ml]
    ax.plot(e, f1, label=f"MaxLen={ml}", color=colors[ml], alpha=0.8)
ax.set_title("Dev F1 Score")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Summary: best dev AUC per max length
ax = axes[2, 0]
best_auc = []
best_auc_epoch = []
for ml in maxlens:
    e, tl, vl, auc, f1 = data[ml]
    idx = np.argmax(auc)
    best_auc.append(auc[idx])
    best_auc_epoch.append(e[idx])

bars = ax.bar([str(ml) for ml in maxlens], best_auc, color=[colors[ml] for ml in maxlens], edgecolor='black', linewidth=0.5)
for bar, a, ep in zip(bars, best_auc, best_auc_epoch):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{a:.4f}\n(ep {ep})", ha='center', va='bottom', fontsize=8)
ax.set_title("Best Dev AUC per Max Length")
ax.set_xlabel("Max Sequence Length")
ax.set_ylabel("Best Dev AUC")
ax.grid(True, alpha=0.3, axis='y')

# Summary: best dev F1 per max length
ax = axes[2, 1]
best_f1 = []
best_f1_epoch = []
for ml in maxlens:
    e, tl, vl, auc, f1 = data[ml]
    idx = np.argmax(f1)
    best_f1.append(f1[idx])
    best_f1_epoch.append(e[idx])

bars = ax.bar([str(ml) for ml in maxlens], best_f1, color=[colors[ml] for ml in maxlens], edgecolor='black', linewidth=0.5)
for bar, f, ep in zip(bars, best_f1, best_f1_epoch):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{f:.4f}\n(ep {ep})", ha='center', va='bottom', fontsize=8)
ax.set_title("Best Dev F1 per Max Length")
ax.set_xlabel("Max Sequence Length")
ax.set_ylabel("Best Dev F1")
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("maxlen_comparison.png", dpi=150, bbox_inches='tight')
print("Saved to maxlen_comparison.png")

# Print summary table
print("\n" + "="*80)
print(f"{'Max Length':>10} | {'Best AUC':>10} | {'AUC Epoch':>9} | {'Best F1':>10} | {'F1 Epoch':>8} | {'Final F1':>10}")
print("-"*80)
for ml in maxlens:
    e, tl, vl, auc, f1 = data[ml]
    auc_idx = np.argmax(auc)
    f1_idx = np.argmax(f1)
    print(f"{ml:>10} | {auc[auc_idx]:>10.4f} | {e[auc_idx]:>9} | {f1[f1_idx]:>10.4f} | {e[f1_idx]:>8} | {f1[-1]:>10.4f}")
