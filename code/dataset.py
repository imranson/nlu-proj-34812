import pandas as pd
import torch
from torch.utils.data import Dataset

class AVDataset(Dataset):
    def __init__(self, data, vocabulary, max_length, lowercase):
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.read_csv(data)
        self.texts_1 = df["text_1"].astype(str).tolist()
        self.texts_2 = df["text_2"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.lowercase = lowercase

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = self.texts_1[idx].lower() if self.lowercase else self.texts_1[idx]
        indices_1 = text1.split()[:self.max_length]
        length_1 = len(indices_1)
        indices_1 = [self.vocabulary.get(word, self.vocabulary["<UNK>"]) for word in indices_1]
        indices_1 = pad_or_truncate(indices_1, self.max_length, self.vocabulary["<PAD>"])
        text2 = self.texts_2[idx].lower() if self.lowercase else self.texts_2[idx]
        indices_2 = text2.split()[:self.max_length]
        length_2 = len(indices_2)
        indices_2 = [self.vocabulary.get(word, self.vocabulary["<UNK>"]) for word in indices_2]
        indices_2 = pad_or_truncate(indices_2, self.max_length, self.vocabulary["<PAD>"])
        return {
            "indices_1": torch.tensor(indices_1, dtype=torch.long),
            "length_1": torch.tensor(length_1, dtype=torch.long),
            "indices_2": torch.tensor(indices_2, dtype=torch.long),
            "length_2": torch.tensor(length_2, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

def pad_or_truncate(sequence, max_length, pad_idx):
    if len(sequence) < max_length:
        padded_seq = sequence + [pad_idx] * (max_length - len(sequence))
    else:
        padded_seq = sequence[:max_length]
    return padded_seq

def build_vocab(texts, lowercase=False):
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    if lowercase:
        texts = [text.lower() for text in texts]
    for text in texts:
        for word in text.split():
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)
    return vocabulary

def load_fasttext_embeddings(embedding_path, vocabulary, embedding_dim):
    embeddings = torch.zeros(len(vocabulary), embedding_dim)
    embedding_indices = []
    with open(embedding_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocabulary:
                vector = torch.tensor(list(map(float, parts[1:])), dtype=torch.float)
                embeddings[vocabulary[word]] = vector
                embedding_indices.append(vocabulary[word])
    embeddings[vocabulary["<UNK>"]] = torch.mean(embeddings[embedding_indices], dim=0)
    return embeddings