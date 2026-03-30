import matplotlib.pyplot as plt
import numpy as np

epochs, train_loss, test_loss, auc, f1 = [], [], [], [], []

with open("slurm-12194579.out") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                e = int(parts[0])
                epochs.append(e)
                train_loss.append(float(parts[1]))
                test_loss.append(float(parts[2]))
                auc.append(float(parts[3]))
                f1.append(float(parts[4]))
            except ValueError:
                continue

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(epochs, train_loss, color='tab:blue')
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')

axes[0, 1].plot(epochs, test_loss, color='tab:orange')
axes[0, 1].set_title('Test Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')

axes[1, 0].plot(epochs, auc, color='tab:green')
axes[1, 0].set_title('AUC')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')

axes[1, 1].plot(epochs, f1, color='tab:red')
axes[1, 1].set_title('F1 Score')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1')

plt.tight_layout()
plt.savefig("metrics_plot.png", dpi=150)
plt.show()
print("Saved to metrics_plot.png")
