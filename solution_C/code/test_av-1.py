import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from config import DEFAULTS

# parse args or use default
parser = argparse.ArgumentParser()
parser.add_argument("--test_csv", type=str, default=DEFAULTS['test_csv'])
parser.add_argument("--model_name", type=str, default=DEFAULTS['model_name'])
parser.add_argument("--model_path", type=str, default=DEFAULTS['model_path'])
parser.add_argument("--batch_size", type=int, default=DEFAULTS['batch_size'])
parser.add_argument("--max_len", type=int, default=DEFAULTS['max_length'])
parser.add_argument("--margin", type=float, default=DEFAULTS['margin'])
parser.add_argument("--dist_lower", type=float, default=DEFAULTS['dist_lower'])
parser.add_argument("--device", type=str, default=DEFAULTS['device'])
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

TEST_CSV = args.test_csv
MODEL_NAME = args.model_name
MODEL_PATH = args.model_path
BATCH_SIZE = args.batch_size
MAX_LEN = args.max_len
MARGIN = args.margin
DIST_LOWER = args.dist_lower
DEVICE = args.device
OUTPUT_PATH = args.output_path

# print parameters
print(f"TEST_CSV={TEST_CSV}, MODEL_NAME={MODEL_NAME}, MODEL_PATH={MODEL_PATH}, BATCH_SIZE={BATCH_SIZE}, MAX_LEN={MAX_LEN}, DIST_LOWER={DIST_LOWER}, MARGIN={MARGIN}, OUTPUT_PATH={OUTPUT_PATH}, DEVICE={DEVICE}")

# load test file
test_pd = pd.read_csv(TEST_CSV)
if 'label' in test_pd.columns:
    test_pd.drop(columns=['label'], inplace=True)

# set up test dataset
class AVDataset(Dataset):
    def __init__(self, text_1, text_2, max_length=MAX_LEN, transform=None, target_transform=None):
        self.text_1 = text_1
        self.text_2 = text_2
        self.transform = transform
        self.target_transform = target_transform
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):
        return len(self.text_1)

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
        }

# load test dataset
test_dataloader = DataLoader(AVDataset(test_pd['text_1'], test_pd['text_2']), batch_size=BATCH_SIZE, shuffle=False)
print(next(iter(test_dataloader))["input_ids_1"].shape)

# set up model
class CustomBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # [CLS] at the beginning for each batch

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2): #calc dist
        emb1 = self.encode(input_ids_1, attention_mask_1)
        emb2 = self.encode(input_ids_2, attention_mask_2)
        return emb1, emb2

device = DEVICE
print(DEVICE)
model = CustomBERT().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
print(model)

sim_fn = nn.CosineSimilarity(dim=-1)
sim_fn.to(device)

# specify float 32 for adequate precision
model.to(torch.float32)

# prediction
model.eval()
test_pred = []
with torch.no_grad():
    for batch, batch_items in enumerate (test_dataloader, 1):
        print(batch*BATCH_SIZE)
        ids1 = batch_items["input_ids_1"].to(device)
        mask1 = batch_items["attention_mask_1"].to(device)
        ids2 = batch_items["input_ids_2"].to(device)
        mask2 = batch_items["attention_mask_2"].to(device)
        embs1, embs2 = model(ids1, mask1, ids2, mask2)
        embs1, embs2 = embs1.squeeze(-1), embs2.squeeze(-1)
        sim = sim_fn(embs1, embs2)
        dist = 1 - sim
        test_pred.extend((dist < MARGIN).tolist())

# save
to_output = test_pd
to_output['label'] = test_pred
to_output['label'] = to_output['label'].astype(int)
to_output.to_csv(OUTPUT_PATH, index=False)