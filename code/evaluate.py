import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
import argparse
    
def compute_metrics(similarities, labels, threshold):
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

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from config import DEFAULTS
    from model import SiameseBiLSTM
    from dataset import build_vocab, load_fasttext_embeddings, AVDataset
    from torch.utils.data import DataLoader

    p = argparse.ArgumentParser(description="Run inference on AV test/dev data")
    p.add_argument("--train_csv", type=str, default=DEFAULTS["train_csv"])
    p.add_argument("--input_csv", type=str, required=True, help="Path to dev or test CSV")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    p.add_argument("--output", type=str, required=True, help="Path to save predictions CSV")
    p.add_argument("--embedding_path", type=str, default=DEFAULTS["embedding_path"])
    p.add_argument("--embedding_dim", type=int, default=DEFAULTS["embedding_dim"])
    p.add_argument("--hidden_dim", type=int, default=DEFAULTS["hidden_dim"])
    p.add_argument("--num_layers", type=int, default=DEFAULTS["num_layers"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--lstm_dropout", type=float, default=DEFAULTS["lstm_dropout"])
    p.add_argument("--freeze_embeddings", action="store_true", default=DEFAULTS["freeze_embeddings"])
    p.add_argument("--lowercase", action="store_true", default=DEFAULTS["lowercase"])
    p.add_argument("--max_length", type=int, default=DEFAULTS["max_length"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--device", type=str, default=DEFAULTS["device"])
    args = p.parse_args()

    device = torch.device(args.device)
 
    train_df = pd.read_csv(args.train_csv)
    vocab = build_vocab(
        train_df["text_1"].astype(str).tolist() + train_df["text_2"].astype(str).tolist(),
        args.lowercase
    )

    embedding_matrix = load_fasttext_embeddings(args.embedding_path, vocab, args.embedding_dim)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    threshold = checkpoint["threshold"]
    print(f"Loaded checkpoint — val F1: {checkpoint['f1']:.4f}, threshold: {threshold:.4f}")

    model = SiameseBiLSTM(
        embedding_matrix,
        args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        lstm_dropout=args.lstm_dropout,
        freeze_embeddings=args.freeze_embeddings
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_df = pd.read_csv(args.input_csv)
    has_labels = "label" in input_df.columns
    if not has_labels:
        input_df["label"] = 0 

    dataset = AVDataset(input_df, vocab, args.max_length, args.lowercase)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_similarities = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["indices_1"].to(device),
                batch["length_1"].to(device),
                batch["indices_2"].to(device),
                batch["length_2"].to(device)
            )
            all_similarities.extend(outputs.squeeze(1).cpu().numpy())

    predictions = (np.array(all_similarities) > threshold).astype(int)
    pd.DataFrame({"prediction": predictions}).to_csv(args.output, index=False)
    print(f"Predictions saved → {args.output} ({len(predictions)} rows)")

    if has_labels:
        labels = np.array(input_df["label"].tolist())
        metrics = compute_metrics(np.array(all_similarities), labels, threshold)
        print("Dev metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))