import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import DEFAULTS
from dataset import AVDataset, build_vocab, load_fasttext_embeddings
from model import SiameseBiLSTM
import pandas as pd
from evaluate import evaluate

def compute_loss(outputs, batch, loss, pos_weight=1.0):
    labels = batch["label"].unsqueeze(1).to(outputs.device).float()
    if loss == "bce":
        weights = torch.where(labels == 1, 
                             torch.tensor(pos_weight, device=outputs.device), 
                             torch.tensor(1.0, device=outputs.device))
        criterion = torch.nn.BCELoss(weight=weights)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(outputs, labels)

def convert_to_serialisable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serialisable(v) for v in obj]
    return obj

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    train_df = pd.read_csv(args.train_csv)
    vocabulary = build_vocab(train_df["text_1"].astype(str).tolist() + train_df["text_2"].astype(str).tolist(), 
                             args.lowercase)

    if args.data_augmentation:
        augmented_df = train_df.copy()
        augmented_df["text_1"], augmented_df["text_2"] = train_df["text_2"].copy(), train_df["text_1"].copy()
        augmented_df["label"] = train_df["label"]
        train_df = pd.concat([train_df, augmented_df], ignore_index=True)
        print(f"Data augmentation enabled. Training samples increased from {len(train_df) // 2} to {len(train_df)}.")
        train_dataset = AVDataset(train_df, vocabulary, args.max_length, args.lowercase)
    else:
        train_dataset = AVDataset(args.train_csv, vocabulary, args.max_length, args.lowercase)
    dev_dataset = AVDataset(args.dev_csv, vocabulary, args.max_length, args.lowercase)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    embeddings = load_fasttext_embeddings(args.embedding_path, vocabulary, args.embedding_dim)

    # Model
    model = SiameseBiLSTM(embeddings, args.hidden_dim, args.dropout, args.num_layers, 
                          args.lstm_dropout, args.freeze_embeddings).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # Training loop
    best_f1 = 0
    history = []
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for step, batch in enumerate(train_loader, 1):
            # Forward pass
            outputs = model(batch["indices_1"].to(device), batch["length_1"].to(device), batch["indices_2"].to(device), batch["length_2"].to(device))
            loss = compute_loss(outputs, batch, args.loss, args.pos_weight)

            # Backward pass
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 200 == 0:
                avg = running_loss / step
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {avg:.4f}")

        # Evaluation
        model.eval()
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, dev_loader, device)
        lr_scheduler.step(val_metrics["f1"])

        history.append({
            "epoch": epoch,
            "train_loss": running_loss / step,
            "train_f1": train_metrics["f1"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_threshold": val_metrics["threshold"],
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": val_metrics["threshold"],
                    "f1": val_metrics["f1"],
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"  ✓ Saved best model (F1={best_f1:.4f}) → {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement in F1. Patience counter: {patience_counter}/5")
            if patience_counter >= 5:
                print("  Early stopping triggered.")
                break

    ckpt_name = (
        f"bilstm_attn_deephead"
        f"_lr{args.lr}"
        f"_hd{args.hidden_dim}"
        f"_do{args.dropout}"
    )

    results_path = os.path.join(args.output_dir, f"{ckpt_name}_results.json")
    results = {
        "best_f1": best_f1,
        "history": history,
        "config": vars(args),
    }
    with open(results_path, "w") as f:
        json.dump(convert_to_serialisable(results), f, indent=2)
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Results saved → {results_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Siamese DeBERTa Authorship Verification")

    p.add_argument("--train_csv", type=str, default=DEFAULTS["train_csv"])
    p.add_argument("--dev_csv", type=str, default=DEFAULTS["dev_csv"])
    p.add_argument("--embedding_path", type=str, default=DEFAULTS["embedding_path"])
    p.add_argument("--max_length", type=int, default=DEFAULTS["max_length"])
    p.add_argument("--data_augmentation", action="store_true", default=DEFAULTS["data_augmentation"])
    p.add_argument("--lowercase", action="store_true", default=DEFAULTS["lowercase"])

    p.add_argument("--hidden_dim", type=int, default=DEFAULTS["hidden_dim"])
    p.add_argument("--embedding_dim", type=int, default=DEFAULTS["embedding_dim"])
    p.add_argument("--num_layers", type=int, default=DEFAULTS["num_layers"])
    p.add_argument("--lstm_dropout", type=float, default=DEFAULTS["lstm_dropout"])
    p.add_argument("--freeze_embeddings", action="store_true", default=DEFAULTS["freeze_embeddings"])
    p.add_argument("--pos_weight", type=float, default=DEFAULTS["pos_weight"])

    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--max_grad_norm", type=float, default=DEFAULTS["max_grad_norm"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--loss", type=str, choices=["bce", "bce_logits"], default=DEFAULTS["loss"])

    p.add_argument("--device", type=str, default=DEFAULTS["device"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Config: {vars(args)}")
    train(args)