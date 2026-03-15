import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
    
def compute_metrics(similarities, labels, threshold):
    # Implement metrics like accuracy, precision, recall, F1
    preds = (similarities > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    accuracy = np.mean(labels == preds)
    auc = roc_auc_score(labels, similarities)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

def find_best_threshold(similarities, labels, num_steps=200):
    sims = np.array(similarities)
    labs = np.array(labels)
    best_f1, best_t = 0, 0
    lo, hi = sims.min(), sims.max()
    for t in np.linspace(lo, hi, num_steps):
        preds = (sims > t).astype(int)
        f1 = f1_score(labs, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def evaluate(model, dataloader, device):
    model.eval()
    all_similarities, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch["indices_1"].to(device), batch["length_1"].to(device), batch["indices_2"].to(device), batch["length_2"].to(device))
            all_similarities.extend(outputs.squeeze(1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
    threshold, _ = find_best_threshold(np.array(all_similarities), np.array(all_labels))
    metrics = compute_metrics(np.array(all_similarities), np.array(all_labels), threshold)
    metrics["threshold"] = threshold
    print("Evaluation results: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
    return metrics