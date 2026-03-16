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

e1, tl1, vl1, auc1, f1_1 = parse_log("logs/slurm-12064421.out")
e2, tl2, vl2, auc2, f1_2 = parse_log("logs/slurm-12064121.out")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DeBERTa-v3-xsmall AV Training: Batch Size 512 vs 1024", fontsize=14, fontweight="bold")

# Training Loss
ax = axes[0, 0]
ax.plot(e1, tl1, label="BS=512")
ax.plot(e2, tl2, label="BS=1024")
ax.set_title("Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# Validation Loss
ax = axes[0, 1]
ax.plot(e1, vl1, label="BS=512")
ax.plot(e2, vl2, label="BS=1024")
ax.set_title("Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# AUC
ax = axes[1, 0]
ax.plot(e1, auc1, label="BS=512")
ax.plot(e2, auc2, label="BS=1024")
ax.set_title("AUC")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.legend()
ax.grid(True, alpha=0.3)

# F1
ax = axes[1, 1]
ax.plot(e1, f1_1, label="BS=512")
ax.plot(e2, f1_2, label="BS=1024")
ax.set_title("F1 Score")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_comparison.png", dpi=150)
plt.show()
print("Saved to training_comparison.png")
