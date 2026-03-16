import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import DEFAULTS

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, default=DEFAULTS['train_csv'])
parser.add_argument("--dev_csv", type=str, default=DEFAULTS['dev_csv'])
parser.add_argument("--model_name", type=str, default=DEFAULTS['model_name'])
parser.add_argument("--batch_size", type=int, default=DEFAULTS['batch_size'])
parser.add_argument("--max_len", type=int, default=DEFAULTS['max_length'])
parser.add_argument("--margin", type=float, default=DEFAULTS['margin'])
parser.add_argument("--dist_lower", type=float, default=DEFAULTS['dist_lower'])
parser.add_argument("--lr", type=float, default=DEFAULTS['lr'])
parser.add_argument("--epochs", type=int, default=DEFAULTS['epochs'])
parser.add_argument("--device", type=str, default=DEFAULTS['device'])
args = parser.parse_args()

TRAIN_CSV = args.train_csv
DEV_CSV = args.dev_csv
MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size
MAX_LEN = args.max_len
MARGIN = args.margin
DIST_LOWER = DEFAULTS['dist_lower']
LR = args.lr
EPOCH = args.epochs
DEVICE = args.device
print(f"TRAIN_CSV={TRAIN_CSV}, DEV_CSV={DEV_CSV}, MODEL_NAME={MODEL_NAME}, BATCH_SIZE={BATCH_SIZE}, MAX_LEN={MAX_LEN}, LR={LR}, EPOCH={EPOCH}, DEVICE={DEVICE}")

train_pd = pd.read_csv(TRAIN_CSV)
train_pd['label']=train_pd['label'].astype(int)
train_pd

class AVDataset(Dataset):
    def __init__(self, text_1, text_2, label, max_length=MAX_LEN, transform=None, target_transform=None):
        self.text_1 = text_1
        self.text_2 = text_2
        self.label = label
        self.transform = transform
        self.target_transform = target_transform
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        seq1 = self.tokenizer(
            self.text_1[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        seq2 = self.tokenizer(
            self.text_2[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids_1": seq1["input_ids"].squeeze(0),
            "attention_mask_1": seq1["attention_mask"].squeeze(0),
            "input_ids_2": seq2["input_ids"].squeeze(0),
            "attention_mask_2": seq2["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label[idx], dtype=torch.float),
        }

train_dataloader = DataLoader(AVDataset(train_pd['text_1'], train_pd['text_2'], train_pd['label']), batch_size=BATCH_SIZE, shuffle=True)
print(next(iter(train_dataloader))["input_ids_1"].shape)

class CustomBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.encoder.config.hidden_size, 1)
        # )

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # [CLS] at the beginning for each batch

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2): #calc dist
        emb1 = self.encode(input_ids_1, attention_mask_1)
        emb2 = self.encode(input_ids_2, attention_mask_2)
        return emb1, emb2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)
model = CustomBERT().to(device)
print(model)

loss_fn = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

model.to(torch.float32)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = label * (dist).pow(2) + (1 - label) * torch.clamp(
            self.margin - dist, min=DIST_LOWER
        ).pow(2)
        return loss.sum()

loss_fn = ContrastiveLoss(MARGIN).to(device)
sim_fn = nn.CosineSimilarity(dim=-1)
sim_fn.to(device)

for x in range(EPOCH):
    train_loss = 0
    model.train()
    for batch, batch_items in enumerate (train_dataloader, 1):
        ids1 = batch_items["input_ids_1"].to(device)
        mask1 = batch_items["attention_mask_1"].to(device)
        ids2 = batch_items["input_ids_2"].to(device)
        mask2 = batch_items["attention_mask_2"].to(device)
        labels = batch_items["label"].to(device)
        embs1, embs2 = model(ids1, mask1, ids2, mask2)
        embs1, embs2 = embs1.squeeze(-1), embs2.squeeze(-1)
        sim = sim_fn(embs1, embs2)
        dist = -1 * sim
        loss = loss_fn(dist, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    dev_loss = 0
    dev_labels, dev_pred = [], []
    dev_pd = pd.read_csv(DEV_CSV)
    dev_pd['label']=dev_pd['label'].astype(int)
    dev_dataloader = DataLoader(AVDataset(dev_pd['text_1'], dev_pd['text_2'], dev_pd['label']), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch, batch_items in enumerate (dev_dataloader, 1):
            ids1 = batch_items["input_ids_1"].to(device)
            mask1 = batch_items["attention_mask_1"].to(device)
            ids2 = batch_items["input_ids_2"].to(device)
            mask2 = batch_items["attention_mask_2"].to(device)
            labels = batch_items["label"].to(device)
            embs1, embs2 = model(ids1, mask1, ids2, mask2)
            embs1, embs2 = embs1.squeeze(-1), embs2.squeeze(-1)
            sim = sim_fn(embs1, embs2)
            dist = -1 * sim
            loss = loss_fn(dist, labels)
            dev_loss += loss.item()
            dev_pred.extend((dist >= MARGIN).tolist())
            dev_labels.extend(labels.tolist())
    print(x+1, train_loss/len(train_pd), dev_loss/len(dev_pd), roc_auc_score(dev_labels, dev_pred), f1_score(dev_labels, dev_pred, average='macro'))







