import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class AVDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length):
        df = pd.read_csv(csv_path)
        self.texts_1 = df["text_1"].astype(str).tolist()
        self.texts_2 = df["text_2"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc1 = self.tokenizer(
            self.texts_1[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            self.texts_2[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids_1": enc1["input_ids"].squeeze(0),
            "attention_mask_1": enc1["attention_mask"].squeeze(0),
            "input_ids_2": enc2["input_ids"].squeeze(0),
            "attention_mask_2": enc2["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class SiameseDeBERTa(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.cos = nn.CosineSimilarity(dim=-1)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # [CLS] at the beginning [0]

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        emb1 = self.encode(input_ids_1, attention_mask_1)
        emb2 = self.encode(input_ids_2, attention_mask_2)
        return self.cos(emb1, emb2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, similarity, label):
        loss = label * (1 - similarity).pow(2) + (1 - label) * torch.clamp(
            similarity - self.margin, min=0
        ).pow(2)
        return loss.mean()

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

def evaluate(model, loader, device):
    model.eval()
    all_similarities, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            sim = model(
                batch["input_ids_1"].to(device),
                batch["attention_mask_1"].to(device),
                batch["input_ids_2"].to(device),
                batch["attention_mask_2"].to(device),
            )
            all_similarities.extend(sim.cpu().tolist())
            all_labels.extend(batch["label"].tolist())
    threshold, f1 = find_best_threshold(all_similarities, all_labels)
    preds = (np.array(all_similarities) > threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, np.array(all_similarities))
    return {"threshold": threshold, "f1": f1, "accuracy": acc, "auc": auc}


def train(args):
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = AVDataset(args.train_csv, tokenizer, args.max_length)
    dev_ds = AVDataset(args.dev_csv, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SiameseDeBERTa(args.model_name).to(device)
    criterion = ContrastiveLoss(margin=args.margin)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    if args.fp16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    best_f1 = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0
        for step, batch in enumerate(train_loader, 1):
            ids1 = batch["input_ids_1"].to(device)
            mask1 = batch["attention_mask_1"].to(device)
            ids2 = batch["input_ids_2"].to(device)
            mask2 = batch["attention_mask_2"].to(device)
            labels = batch["label"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    dist = model(ids1, mask1, ids2, mask2)
                    loss = criterion(dist, labels)
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                dist = model(ids1, mask1, ids2, mask2)
                loss = criterion(dist, labels)
                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            if step % args.log_steps == 0:
                avg = running_loss / step
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {avg:.4f}")

        # Validation
        metrics = evaluate(model, dev_loader, device)
        print(
            f"Epoch {epoch} — Val F1: {metrics['f1']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f} | "
            f"Threshold: {metrics['threshold']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": metrics["threshold"],
                    "f1": metrics["f1"],
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"  ✓ Saved best model (F1={best_f1:.4f}) → {save_path}")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")

def parse_args():
    p = argparse.ArgumentParser(description="Siamese DeBERTa Authorship Verification")

    p.add_argument("--train_csv", type=str, default="training_data/AV/train.csv")
    p.add_argument("--dev_csv", type=str, default="training_data/AV/dev.csv")

    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    p.add_argument("--max_length", type=int, default=512)

    p.add_argument("--margin", type=float, default=-0.5)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_steps", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="checkpoints/av_siamese")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Config: {vars(args)}")
    train(args)
