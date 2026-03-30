import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score

from config import DEFAULTS

# parse args or use default
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
parser.add_argument("--output_dir", type=str, default=DEFAULTS['output_dir'])
parser.add_argument("--save_model", action="store_true", default=DEFAULTS['save_model'])
args = parser.parse_args()

TRAIN_CSV = args.train_csv
DEV_CSV = args.dev_csv
MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size
MAX_LEN = args.max_len
MARGIN = args.margin
DIST_LOWER = args.dist_lower
LR = args.lr
EPOCH = args.epochs
DEVICE = args.device
OUTPUT_DIR = args.output_dir
SAVE_MODEL = args.save_model
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_short = MODEL_NAME.split("/")[-1]
RUN_ID = f"{model_short}_ep{EPOCH}_bs{BATCH_SIZE}_lr{LR}_ml{MAX_LEN}_m{MARGIN}_dl{DIST_LOWER}"

# print parameters
print(f"TRAIN_CSV={TRAIN_CSV}, DEV_CSV={DEV_CSV}, MODEL_NAME={MODEL_NAME}, MARGIN={MARGIN}, DIST_LOWER={DIST_LOWER}, BATCH_SIZE={BATCH_SIZE}, MAX_LEN={MAX_LEN}, LR={LR}, EPOCH={EPOCH}, DEVICE={DEVICE}")

# load train file
train_pd = pd.read_csv(TRAIN_CSV)
train_pd['label']=train_pd['label'].astype(int)
train_pd

# set up train dataset
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

# load train dataset
train_dataloader = DataLoader(AVDataset(train_pd['text_1'], train_pd['text_2'], train_pd['label']), batch_size=BATCH_SIZE, shuffle=True)
print(next(iter(train_dataloader))["input_ids_1"].shape)

# set up model
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

# specify float 32 for adequate precision
model.to(torch.float32)

# contrastive loss
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

# history tracking for model vis and selection
train_hist = []
test_hist = []
auc_hist = []
f1_hist = []
best_auc = 0.0
for x in range(EPOCH):
    # training loop
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
        dist = 1 - sim
        loss = loss_fn(dist, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # eval loop
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
            dist = 1 - sim
            loss = loss_fn(dist, labels)
            dev_loss += loss.item()
            dev_pred.extend((dist < MARGIN).tolist())
            dev_labels.extend(labels.tolist())

    # tracking and saving
    train_hist.append(train_loss/len(train_pd))
    test_hist.append(dev_loss/len(dev_pd))
    auc_hist.append(roc_auc_score(dev_labels, dev_pred))
    f1_hist.append(f1_score(dev_labels, dev_pred, average='macro'))
    print(x+1, train_hist[-1], test_hist[-1],auc_hist[-1],f1_hist[-1])
    if SAVE_MODEL and auc_hist[-1] > best_auc:
        best_auc = auc_hist[-1]
        model_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}_model.pt")
        torch.save(model.state_dict(), model_path)

# make table for reference
history_df = pd.DataFrame({
    "epoch": list(range(1, EPOCH + 1)),
    "train_loss": train_hist,
    "dev_loss": test_hist,
    "auc": auc_hist,
    "f1": f1_hist,
})
history_df["best_auc"] = history_df["auc"] == history_df["auc"].max()
history_df.to_csv(os.path.join(OUTPUT_DIR, f"{RUN_ID}_history.csv"), index=False)

# make graph and save
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
epochs = history_df["epoch"]
ax1.plot(epochs, history_df["train_loss"], label="Train Loss")
ax1.plot(epochs, history_df["dev_loss"], label="Dev Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss")
ax1.legend()
ax2.plot(epochs, history_df["auc"], label="AUC")
ax2.plot(epochs, history_df["f1"], label="F1")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.set_title("Metrics")
best_idx = history_df["auc"].idxmax()
best_epoch = int(history_df.loc[best_idx, "epoch"])
best_auc_val = history_df["auc"].max()
ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
ax2.annotate(f"Best AUC: {best_auc_val:.4f} (ep {best_epoch})",
             xy=(best_epoch, best_auc_val), xytext=(5, -15),
             textcoords="offset points", fontsize=9,
             arrowprops=dict(arrowstyle="->", color="gray"))
ax2.legend()
fig.suptitle(RUN_ID)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"{RUN_ID}_history.png"), dpi=150)
print(f"Saved outputs to {OUTPUT_DIR}/ with prefix {RUN_ID}")

