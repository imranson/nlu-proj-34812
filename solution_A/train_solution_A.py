"""
Solution A: Traditional ML Approach for Authorship Verification
================================================================
This script implements a stylometric feature-based approach using a
Gradient Boosting classifier for the Authorship Verification (AV) shared task.

Approach Overview:
- Extract stylometric (writing style) features from each text in a pair
- Compute pairwise difference/similarity features between the two texts
- Use TF-IDF character n-gram cosine similarity as a key feature
- Apply stylometric distance measures (NCD, Burrows' Delta, Keselj dissimilarity)
- Train a Gradient Boosting classifier on these features

Feature Categories:
1. Character-level: punctuation ratios, character class distributions
2. Lexical: vocabulary richness (TTR, hapax, Yule's K), word length distributions,
   function word frequencies, contraction usage, stopword ratios
3. Structural: sentence lengths, text length ratios, sentence length variance
4. Similarity: TF-IDF char/word n-gram cosine similarity, Jaccard word overlap
5. Stylometric distances: Normalised Compression Distance (NCD),
   Burrows' Delta, Keselj character n-gram profile dissimilarity

References:
- Stamatatos (2009). "A survey of modern authorship attribution methods"
- Keselj et al. (2003). "N-gram-based author profiles for authorship attribution"
- Koppel & Winter (2014). "Determining if two documents are written by the same author"
- Burrows (2002). "Delta: a measure of stylistic difference and a guide to likely authorship"
- Cilibrasi & Vitanyi (2005). "Clustering by compression"
"""

import pandas as pd
import numpy as np
import re
import zlib
import pickle
import os
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ============================================================

def vocabulary_richness(text):
    """
    Compute vocabulary richness measures for a text.
    TTR and hapax ratio capture an author's lexical diversity.
    """
    words = text.lower().split()
    if not words:
        return 0.0, 0.0, 0.0

    total_words = len(words)
    unique_words = set(words)
    ttr = len(unique_words) / total_words
    word_counts = Counter(words)
    hapax = sum(1 for count in word_counts.values() if count == 1)
    hapax_ratio = hapax / total_words
    avg_word_len = np.mean([len(w) for w in words])

    return ttr, hapax_ratio, avg_word_len


def yules_k(words):
    """
    Yule's K measure of vocabulary richness.
    Higher K indicates less diverse vocabulary. This captures an
    author's tendency towards repetition vs. lexical variety.
    """
    if not words:
        return 0.0
    freq_dist = Counter(words)
    N = len(words)
    M = sum(v * v for v in freq_dist.values())
    return 10000 * (M - N) / (N * N) if N > 1 else 0.0


def punctuation_features(text):
    """
    Extract punctuation usage features.
    Punctuation habits are a strong stylometric signal - they reflect
    an author's unconscious writing habits.
    """
    total_chars = len(text) if text else 1

    features = {
        'comma_rate': text.count(',') / total_chars,
        'period_rate': text.count('.') / total_chars,
        'excl_rate': text.count('!') / total_chars,
        'quest_rate': text.count('?') / total_chars,
        'semicolon_rate': text.count(';') / total_chars,
        'colon_rate': text.count(':') / total_chars,
        'dash_rate': text.count('-') / total_chars,
        'quote_rate': (text.count('"') + text.count("'")) / total_chars,
        'paren_rate': (text.count('(') + text.count(')')) / total_chars,
        'ellipsis_rate': text.count('...') / total_chars,
    }
    return features


def function_word_features(text):
    """
    Extract function word frequency features.
    Function words (the, a, is, etc.) are content-independent and reflect
    an author's unconscious syntactic preferences.
    """
    function_words = [
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'not', 'no', 'so', 'yet', 'just', 'also', 'now', 'then',
        'all', 'any', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
        'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
        'about', 'after', 'before', 'into', 'through', 'up', 'down',
        'out', 'off', 'over', 'because', 'while', 'since',
    ]

    words = text.lower().split()
    total = len(words) if words else 1

    features = {}
    for fw in function_words:
        features[f'fw_{fw}'] = words.count(fw) / total

    return features


def text_structure_features(text):
    """
    Extract structural features about the text's composition.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()
    total_words = len(words) if words else 1
    total_chars = len(text) if text else 1

    sent_lengths = [len(s.split()) for s in sentences] if sentences else [0]
    word_lengths = [len(w) for w in words] if words else [0]

    short_words = sum(1 for wl in word_lengths if wl <= 3) / total_words
    medium_words = sum(1 for wl in word_lengths if 4 <= wl <= 6) / total_words
    long_words = sum(1 for wl in word_lengths if wl >= 7) / total_words

    upper_chars = sum(1 for c in text if c.isupper())
    digit_chars = sum(1 for c in text if c.isdigit())

    features = {
        'avg_sent_len': np.mean(sent_lengths),
        'std_sent_len': np.std(sent_lengths) if len(sent_lengths) > 1 else 0,
        'num_sentences': len(sentences),
        'avg_word_len': np.mean(word_lengths),
        'std_word_len': np.std(word_lengths) if len(word_lengths) > 1 else 0,
        'short_word_ratio': short_words,
        'medium_word_ratio': medium_words,
        'long_word_ratio': long_words,
        'upper_ratio': upper_chars / total_chars,
        'digit_ratio': digit_chars / total_chars,
        'space_ratio': text.count(' ') / total_chars,
        'total_words': total_words,
        'total_chars': total_chars,
    }
    return features


def extract_single_text_features(text):
    """Extract all stylometric features for a single text."""
    features = {}

    ttr, hapax_ratio, avg_wl = vocabulary_richness(text)
    features['ttr'] = ttr
    features['hapax_ratio'] = hapax_ratio
    features['vocab_avg_word_len'] = avg_wl

    punct_feats = punctuation_features(text)
    features.update(punct_feats)

    fw_feats = function_word_features(text)
    features.update(fw_feats)

    struct_feats = text_structure_features(text)
    features.update(struct_feats)

    return features


# ============================================================
# 2. STYLOMETRIC DISTANCE MEASURES
# ============================================================

def normalised_compression_distance(text1, text2):
    """
    Normalised Compression Distance (NCD) from Cilibrasi & Vitanyi (2005).

    NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

    where C(x) is the compressed size of x. Lower NCD means texts are
    more similar in their statistical properties, capturing stylistic
    similarity without any explicit feature engineering.
    """
    b1 = text1.encode('utf-8')
    b2 = text2.encode('utf-8')
    c1 = len(zlib.compress(b1))
    c2 = len(zlib.compress(b2))
    c12 = len(zlib.compress(b1 + b2))
    max_c = max(c1, c2)
    min_c = min(c1, c2)
    if max_c == 0:
        return 0.0
    return (c12 - min_c) / max_c


def burrows_delta(text1, text2):
    """
    Simplified Burrows' Delta from Burrows (2002).

    Computes the mean absolute difference of word frequency distributions.
    Lower delta indicates more similar writing styles. This is one of the
    most widely used measures in computational stylometry.
    """
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    if not words1 or not words2:
        return 0.0

    freq1 = {w: c / len(words1) for w, c in Counter(words1).items()}
    freq2 = {w: c / len(words2) for w, c in Counter(words2).items()}
    all_words = set(freq1.keys()) | set(freq2.keys())

    if not all_words:
        return 0.0

    return sum(abs(freq1.get(w, 0) - freq2.get(w, 0)) for w in all_words) / len(all_words)


def keselj_dissimilarity(text1, text2, n=3, L=500):
    """
    Character n-gram profile dissimilarity from Keselj et al. (2003).

    For each text, builds a profile of the L most frequent character n-grams.
    Computes dissimilarity as the sum of squared normalised frequency differences.
    Lower dissimilarity indicates more similar writing style.
    """
    text1_lower = text1.lower()
    text2_lower = text2.lower()

    ngrams1 = Counter(text1_lower[i:i+n] for i in range(len(text1_lower) - n + 1))
    ngrams2 = Counter(text2_lower[i:i+n] for i in range(len(text2_lower) - n + 1))

    total1 = sum(ngrams1.values()) or 1
    total2 = sum(ngrams2.values()) or 1

    profile1 = {ng: c / total1 for ng, c in ngrams1.most_common(L)}
    profile2 = {ng: c / total2 for ng, c in ngrams2.most_common(L)}

    all_ngrams = set(profile1.keys()) | set(profile2.keys())

    if not all_ngrams:
        return 0.0

    dissimilarity = 0.0
    for ng in all_ngrams:
        f1 = profile1.get(ng, 0)
        f2 = profile2.get(ng, 0)
        avg = (f1 + f2) / 2
        if avg > 0:
            dissimilarity += ((f1 - f2) / avg) ** 2

    return dissimilarity


# ============================================================
# 3. PAIRWISE FEATURE EXTRACTION
# ============================================================

# Common stopwords set (module-level for efficiency)
STOPWORDS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'i', 'me', 'my', 'we', 'us', 'you', 'your', 'he', 'him',
    'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this',
    'that', 'not', 'no'
})

CONTRACTIONS = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]


def extract_pair_features(text1, text2):
    """
    Extract features for a pair of texts.

    Computes:
    1. Absolute differences of individual stylometric features
    2. Pairwise similarity/distance metrics
    3. Stylometric distance measures from the literature
    """
    feats1 = extract_single_text_features(text1)
    feats2 = extract_single_text_features(text2)

    pair_features = {}

    # Absolute differences of all individual features
    for key in feats1:
        pair_features[f'diff_{key}'] = abs(feats1[key] - feats2[key])

    # Word-level Jaccard similarity
    words1_set = set(text1.lower().split())
    words2_set = set(text2.lower().split())
    union = words1_set | words2_set
    pair_features['word_jaccard'] = len(words1_set & words2_set) / len(union) if union else 0.0

    # Word bigram Jaccard
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    bigrams1 = set(zip(words1[:-1], words1[1:])) if len(words1) > 1 else set()
    bigrams2 = set(zip(words2[:-1], words2[1:])) if len(words2) > 1 else set()
    if bigrams1 | bigrams2:
        pair_features['word_bigram_jaccard'] = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
    else:
        pair_features['word_bigram_jaccard'] = 0.0

    # Length ratio features
    len1, len2 = len(text1), len(text2)
    pair_features['char_len_ratio'] = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

    wc1, wc2 = len(words1), len(words2)
    pair_features['word_count_ratio'] = min(wc1, wc2) / max(wc1, wc2) if max(wc1, wc2) > 0 else 1.0

    # --- Stylometric distance measures ---

    # Normalised Compression Distance
    pair_features['ncd'] = normalised_compression_distance(text1, text2)

    # Burrows' Delta
    pair_features['burrows_delta'] = burrows_delta(text1, text2)

    # Keselj character n-gram profile dissimilarity
    pair_features['keselj_3gram'] = keselj_dissimilarity(text1, text2, n=3, L=500)
    pair_features['keselj_4gram'] = keselj_dissimilarity(text1, text2, n=4, L=500)

    # --- Additional stylometric features ---

    # Yule's K vocabulary richness difference
    pair_features['diff_yules_k'] = abs(yules_k(words1) - yules_k(words2))

    # Contraction usage difference
    c1 = sum(text1.lower().count(c) for c in CONTRACTIONS) / (wc1 or 1)
    c2 = sum(text2.lower().count(c) for c in CONTRACTIONS) / (wc2 or 1)
    pair_features['diff_contraction_rate'] = abs(c1 - c2)

    # Stopword ratio difference
    sr1 = sum(1 for w in words1 if w in STOPWORDS) / (wc1 or 1)
    sr2 = sum(1 for w in words2 if w in STOPWORDS) / (wc2 or 1)
    pair_features['diff_stopword_ratio'] = abs(sr1 - sr2)

    # Sentence length variance difference
    sents1 = [s for s in re.split(r'[.!?]+', text1) if s.strip()]
    sents2 = [s for s in re.split(r'[.!?]+', text2) if s.strip()]
    sl1 = [len(s.split()) for s in sents1] if sents1 else [0]
    sl2 = [len(s.split()) for s in sents2] if sents2 else [0]
    pair_features['diff_sent_len_var'] = abs(np.var(sl1) - np.var(sl2))

    return pair_features


# ============================================================
# 4. DATA LOADING AND FEATURE MATRIX CONSTRUCTION
# ============================================================

def load_data(filepath):
    """Load a CSV file and return text pairs and labels."""
    df = pd.read_csv(filepath)
    texts1 = df['text_1'].fillna('').tolist()
    texts2 = df['text_2'].fillna('').tolist()
    labels = df['label'].astype(int).tolist() if 'label' in df.columns else None
    return texts1, texts2, labels


def build_feature_matrix(texts1, texts2, char_vectorizer=None, word_vectorizer=None,
                         fit=True, verbose=True):
    """
    Build the complete feature matrix combining:
    1. Stylometric difference features
    2. Stylometric distance measures (NCD, Burrows' Delta, Keselj)
    3. TF-IDF character n-gram cosine similarity
    4. TF-IDF word n-gram cosine similarity
    """
    n = len(texts1)

    # --- Stylometric features ---
    if verbose:
        print("  Extracting stylometric features...")
    all_features = []
    for i in range(n):
        if verbose and (i + 1) % 5000 == 0:
            print(f"    Pair {i+1}/{n}...")
        feats = extract_pair_features(texts1[i], texts2[i])
        all_features.append(feats)

    feature_df = pd.DataFrame(all_features)

    # --- TF-IDF character n-gram cosine similarity ---
    if verbose:
        print("  Computing TF-IDF char n-gram similarities...")

    all_texts = texts1 + texts2

    if fit:
        char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',       # Character n-grams with word boundaries
            ngram_range=(2, 5),       # Bigrams to 5-grams
            max_features=10000,       # Limit vocabulary size to prevent overfitting
            sublinear_tf=True,        # Apply sublinear TF scaling (1 + log(tf))
            min_df=5,                 # Ignore very rare n-grams
        )
        char_vectorizer.fit(all_texts)

    # Efficient sparse cosine similarity computation
    vecs1_char = normalize(char_vectorizer.transform(texts1), norm='l2')
    vecs2_char = normalize(char_vectorizer.transform(texts2), norm='l2')
    char_sims = np.array(vecs1_char.multiply(vecs2_char).sum(axis=1)).flatten()

    # --- TF-IDF word n-gram cosine similarity ---
    if verbose:
        print("  Computing TF-IDF word n-gram similarities...")

    if fit:
        word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=3,
        )
        word_vectorizer.fit(all_texts)

    vecs1_word = normalize(word_vectorizer.transform(texts1), norm='l2')
    vecs2_word = normalize(word_vectorizer.transform(texts2), norm='l2')
    word_sims = np.array(vecs1_word.multiply(vecs2_word).sum(axis=1)).flatten()

    # --- Combine all features ---
    feature_df['tfidf_char_cosine'] = char_sims
    feature_df['tfidf_word_cosine'] = word_sims

    all_feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, all_feature_names, char_vectorizer, word_vectorizer


# ============================================================
# 5. TRAINING
# ============================================================

def train_model(X_train, y_train):
    """
    Train classifiers with feature scaling.

    We train both a Logistic Regression (for interpretability) and a
    Gradient Boosting classifier (for best performance), then select
    the one with better dev performance.
    """
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training Logistic Regression classifier...")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)

    print("Training Gradient Boosting classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=350,          # Number of boosting rounds
        max_depth=5,               # Tree depth
        learning_rate=0.1,         # Shrinkage rate
        subsample=0.8,             # Row subsampling for regularisation
        min_samples_leaf=25,       # Minimum samples per leaf (prevents overfitting)
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train)

    print("Training complete.")
    return lr_model, gb_model, scaler


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(model, scaler, X, y_true, dataset_name="Dev", model_name="Model"):
    """Evaluate the model and print comprehensive metrics."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_binary = f1_score(y_true, y_pred, average='binary')
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')

    print(f"\n{'='*55}")
    print(f"  {model_name} - {dataset_name} Set Evaluation")
    print(f"{'='*55}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  F1 (binary):       {f1_binary:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro):    {rec:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Different Author (0)', 'Same Author (1)']))
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    return y_pred, {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_binary': f1_binary,
        'precision_macro': prec,
        'recall_macro': rec,
    }


# ============================================================
# 7. MAIN TRAINING PIPELINE
# ============================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'training_data', 'AV')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Load data ---
    print("Loading training data...")
    train_texts1, train_texts2, train_labels = load_data(os.path.join(DATA_DIR, 'train.csv'))
    print(f"  Training pairs: {len(train_texts1)}")
    print(f"  Label distribution: 0={train_labels.count(0)}, 1={train_labels.count(1)}")

    print("\nLoading dev data...")
    dev_texts1, dev_texts2, dev_labels = load_data(os.path.join(DATA_DIR, 'dev.csv'))
    print(f"  Dev pairs: {len(dev_texts1)}")
    print(f"  Label distribution: 0={dev_labels.count(0)}, 1={dev_labels.count(1)}")

    # --- Extract features ---
    print("\nExtracting training features...")
    X_train, feature_names, char_vec, word_vec = build_feature_matrix(
        train_texts1, train_texts2, fit=True
    )
    y_train = np.array(train_labels)
    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Number of features: {len(feature_names)}")

    print("\nExtracting dev features...")
    X_dev, _, _, _ = build_feature_matrix(
        dev_texts1, dev_texts2,
        char_vectorizer=char_vec,
        word_vectorizer=word_vec,
        fit=False
    )
    y_dev = np.array(dev_labels)

    # --- Train models ---
    print("\n" + "="*55)
    print("  TRAINING MODELS")
    print("="*55)
    lr_model, gb_model, scaler = train_model(X_train, y_train)

    # --- Evaluate both models ---
    print("\n" + "="*55)
    print("  LOGISTIC REGRESSION RESULTS")
    print("="*55)
    evaluate_model(lr_model, scaler, X_train, y_train, "Training", "Logistic Regression")
    _, lr_dev_metrics = evaluate_model(lr_model, scaler, X_dev, y_dev, "Dev", "Logistic Regression")

    print("\n" + "="*55)
    print("  GRADIENT BOOSTING RESULTS")
    print("="*55)
    evaluate_model(gb_model, scaler, X_train, y_train, "Training", "Gradient Boosting")
    _, gb_dev_metrics = evaluate_model(gb_model, scaler, X_dev, y_dev, "Dev", "Gradient Boosting")

    # --- Select best model ---
    if gb_dev_metrics['f1_macro'] > lr_dev_metrics['f1_macro']:
        best_model = gb_model
        best_name = "Gradient Boosting"
        best_metrics = gb_dev_metrics
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_metrics = lr_dev_metrics

    print(f"\n*** Best model: {best_name} (Dev F1 macro = {best_metrics['f1_macro']:.4f}) ***")

    # --- Save model artefacts ---
    print("\nSaving model artefacts...")

    with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODEL_DIR, 'char_vectorizer.pkl'), 'wb') as f:
        pickle.dump(char_vec, f)

    with open(os.path.join(MODEL_DIR, 'word_vectorizer.pkl'), 'wb') as f:
        pickle.dump(word_vec, f)

    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    print(f"  All artefacts saved to: {MODEL_DIR}/")

    # --- Feature importance analysis ---
    print("\n--- Feature Importance (Gradient Boosting) ---")
    importances = gb_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]

    print("\nTop 20 most important features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:35s} (importance = {importances[idx]:.4f})")

    # --- Feature importance analysis (LR for interpretability) ---
    print("\n--- Feature Importance (Logistic Regression Coefficients) ---")
    importance = np.abs(lr_model.coef_[0])
    top_indices = np.argsort(importance)[::-1][:20]

    print("\nTop 20 most important features:")
    for rank, idx in enumerate(top_indices, 1):
        coef = lr_model.coef_[0][idx]
        direction = "+" if coef > 0 else "-"
        print(f"  {rank:2d}. {feature_names[idx]:35s} ({direction} |coef| = {importance[idx]:.4f})")

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
