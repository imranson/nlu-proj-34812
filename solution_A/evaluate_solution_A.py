"""
Solution A: Evaluation Script
==============================
This script evaluates the trained traditional ML model on the dev set,
producing comprehensive metrics including F1-score, precision, recall,
confusion matrix, and per-class performance.

Usage:
    python evaluate_solution_A.py
"""

import pickle
import numpy as np
import os
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, precision_score, recall_score
)

# Import feature extraction from training module
from train_solution_A import load_data, build_feature_matrix

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, '..', 'training_data', 'AV')

    # --- Load saved model artefacts ---
    print("Loading model artefacts...")
    with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'char_vectorizer.pkl'), 'rb') as f:
        char_vec = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'word_vectorizer.pkl'), 'rb') as f:
        word_vec = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)

    print(f"  Model type: {type(model).__name__}")
    print(f"  Number of features: {len(feature_names)}")

    # --- Load dev data ---
    print("\nLoading dev data...")
    dev_texts1, dev_texts2, dev_labels = load_data(os.path.join(DATA_DIR, 'dev.csv'))
    print(f"  Dev pairs: {len(dev_texts1)}")
    print(f"  Label distribution: 0={dev_labels.count(0)}, 1={dev_labels.count(1)}")

    # --- Extract features ---
    print("\nExtracting features from dev set...")
    X_dev, _, _, _ = build_feature_matrix(
        dev_texts1, dev_texts2,
        char_vectorizer=char_vec,
        word_vectorizer=word_vec,
        fit=False
    )
    y_dev = np.array(dev_labels)

    # --- Generate predictions ---
    print("\nGenerating predictions...")
    X_dev_scaled = scaler.transform(X_dev)
    y_pred = model.predict(X_dev_scaled)

    # --- Compute metrics ---
    acc = accuracy_score(y_dev, y_pred)
    f1_macro = f1_score(y_dev, y_pred, average='macro')
    f1_weighted = f1_score(y_dev, y_pred, average='weighted')
    f1_binary = f1_score(y_dev, y_pred, average='binary')
    prec_macro = precision_score(y_dev, y_pred, average='macro')
    rec_macro = recall_score(y_dev, y_pred, average='macro')

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS ON DEV SET")
    print(f"{'='*60}")
    print(f"  Accuracy:               {acc:.4f}")
    print(f"  F1-score (macro):       {f1_macro:.4f}")
    print(f"  F1-score (weighted):    {f1_weighted:.4f}")
    print(f"  F1-score (binary):      {f1_binary:.4f}")
    print(f"  Precision (macro):      {prec_macro:.4f}")
    print(f"  Recall (macro):         {rec_macro:.4f}")

    print(f"\n  Per-Class Classification Report:")
    print(classification_report(
        y_dev, y_pred,
        target_names=['Different Author (0)', 'Same Author (1)'],
        digits=4
    ))

    cm = confusion_matrix(y_dev, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                     Predicted 0    Predicted 1")
    print(f"    Actual 0 (Diff)    {cm[0][0]:>8d}       {cm[0][1]:>8d}")
    print(f"    Actual 1 (Same)    {cm[1][0]:>8d}       {cm[1][1]:>8d}")

    # --- Error analysis ---
    print(f"\n{'='*60}")
    print(f"  BRIEF ERROR ANALYSIS")
    print(f"{'='*60}")

    # Identify misclassified examples
    false_positives = np.where((y_pred == 1) & (y_dev == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y_dev == 1))[0]

    print(f"\n  False Positives (predicted same, actually different): {len(false_positives)}")
    print(f"  False Negatives (predicted different, actually same):  {len(false_negatives)}")

    # Analyse feature distributions for errors
    print(f"\n  Mean TF-IDF char cosine similarity:")
    tfidf_char_idx = feature_names.index('tfidf_char_cosine')
    print(f"    Correct predictions:     {X_dev[y_pred == y_dev, tfidf_char_idx].mean():.4f}")
    print(f"    False Positives:         {X_dev[false_positives, tfidf_char_idx].mean():.4f}")
    print(f"    False Negatives:         {X_dev[false_negatives, tfidf_char_idx].mean():.4f}")

    print(f"\n  Mean word Jaccard similarity:")
    jaccard_idx = feature_names.index('word_jaccard')
    print(f"    Correct predictions:     {X_dev[y_pred == y_dev, jaccard_idx].mean():.4f}")
    print(f"    False Positives:         {X_dev[false_positives, jaccard_idx].mean():.4f}")
    print(f"    False Negatives:         {X_dev[false_negatives, jaccard_idx].mean():.4f}")

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
