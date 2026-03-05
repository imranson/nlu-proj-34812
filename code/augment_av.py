import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Augment AV data by swapping text pairs.")
parser.add_argument("--input", required=True, help="Path to input CSV")
parser.add_argument("--output", required=True, help="Path to output CSV")
args = parser.parse_args()

df = pd.read_csv(args.input)
swapped = df.rename(columns={"text_1": "text_2", "text_2": "text_1"})[df.columns]
augmented = pd.concat([df, swapped], ignore_index=True)
augmented = augmented.drop_duplicates(subset=["text_1", "text_2", "label"])
augmented.to_csv(args.output, index=False)

print(f"Original: {len(df)} rows")
print(f"Augmented: {len(augmented)} rows")
print(f"Saved to {args.output}")
