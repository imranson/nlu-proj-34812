import pandas as pd
from collections import Counter

TRAIN_PATH = "./training_data/AV/train.csv"
DEV_PATH = "./training_data/AV/dev.csv"

train = pd.read_csv(TRAIN_PATH)
dev = pd.read_csv(DEV_PATH)

print(f"Train: {len(train)} rows | Dev: {len(dev)} rows\n")

for name, df in [("train", train), ("dev", dev)]:
    texts_1 = set(df["text_1"])
    texts_2 = set(df["text_2"])
    overlap = texts_1 & texts_2
    print(f"[{name}] Texts appearing in BOTH text_1 and text_2 columns: {len(overlap)}")

for name, df in [("train", train), ("dev", dev)]:
    for col in ["text_1", "text_2"]:
        counts = df[col].value_counts()
        repeats = counts[counts > 1]
        print(f"[{name}] Repeated texts in '{col}': {len(repeats)} unique texts "
              f"(total extra occurrences: {repeats.sum() - len(repeats)})")

all_train_texts = set(train["text_1"]) | set(train["text_2"])
all_dev_texts = set(dev["text_1"]) | set(dev["text_2"])
train_dev_overlap = all_train_texts & all_dev_texts
print(f"\nTexts shared between train and dev: {len(train_dev_overlap)}")

all_texts = list(train["text_1"]) + list(train["text_2"]) + list(dev["text_1"]) + list(dev["text_2"])
text_counts = Counter(all_texts)
total = len(all_texts)
unique = len(text_counts)
print(f"\nTotal text slots (all rows × 2 columns, train+dev): {total}")
print(f"Unique texts: {unique}")
print(f"Texts appearing more than once: {sum(1 for c in text_counts.values() if c > 1)}")

most_common = text_counts.most_common(10)
print(f"\nTop 10 most repeated texts (count | first 80 chars):")
for text, count in most_common:
    preview = text[:80].replace("\n", " ")
    print(f"  {count}x | {preview}...")
