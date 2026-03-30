---
language: en
license: cc-by-4.0
tags:
- text-classification
- authorship-verification
repo: https://github.com/username/project_name
---

# Model Card for m13447tk-q12628ct-AV

<!-- Provide a quick summary of what the model is/does. -->

A traditional machine learning model for Authorship Verification (AV) that determines whether two text sequences were written by the same author, using handcrafted stylometric features with a Gradient Boosting classifier.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model extracts stylometric (writing style) features from pairs of texts, including punctuation usage rates, vocabulary richness measures (type-token ratio, hapax legomena, Yule's K), function word frequencies, text structure statistics (sentence/word length distributions), and pairwise similarity measures (Normalised Compression Distance, Burrows' Delta, Keselj character n-gram dissimilarity). TF-IDF cosine similarities from character n-grams (2-5) and word n-grams (1-2) are also computed. The resulting feature vector is scaled with StandardScaler and classified by a Gradient Boosting model.

- **Developed by:** Timothy Kirathe and Tham Chun Shen
- **Language(s):** English
- **Model type:** Supervised (Traditional Machine Learning)
- **Model architecture:** Gradient Boosting Classifier with stylometric feature engineering
- **Finetuned from model [optional]:** N/A (not fine-tuned from a pre-trained model)

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** N/A (trained from scratch)
- **Paper or documentation:** Stamatatos (2009), Keselj et al. (2003), Koppel & Winter (2014), Burrows (2002), Cilibrasi & Vitanyi (2005)

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

Approximately 27,643 pairs of text sequences from the COMP34812 AV shared task (closed track). Texts include emails, news articles, and blog posts. Labels are binary: 0 (different author) or 1 (same author).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - n_estimators: 350
      - max_depth: 5
      - learning_rate: 0.1
      - subsample: 0.8
      - min_samples_leaf: 25
      - random_state: 42
      - StandardScaler applied to all features
      - TF-IDF char n-grams: analyzer='char_wb', ngram_range=(2,5), max_features=10000, sublinear_tf=True, min_df=5
      - TF-IDF word n-grams: analyzer='word', ngram_range=(1,2), max_features=5000, sublinear_tf=True, min_df=3

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - Overall training time: approximately 10-15 minutes on CPU
      - Model size: 1.6 MB total (5 artefact files)
      - No GPU required

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Approximately 5,993 pairs from the COMP34812 AV development set.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - F1-score (macro, weighted)
      - Precision (macro)
      - Recall (macro)

### Results

The model achieved the following on the development set:
      - Accuracy: 0.6905
      - F1 (macro): 0.6901
      - F1 (weighted): 0.6898
      - Precision (macro): 0.6931
      - Recall (macro): 0.6915
      
      Per-class performance:
      - Different Author (0): Precision 0.6651, Recall 0.7419, F1 0.7014
      - Same Author (1): Precision 0.7210, Recall 0.6410, F1 0.6787

## Technical Specifications

### Hardware


      - RAM: 4 GB minimum
      - Storage: less than 10 MB for model artefacts
      - GPU: not required (CPU-only)

### Software


      - Python 3.10+
      - scikit-learn
      - pandas
      - numpy

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- The model relies on surface-level stylometric features and may be deceived by intentional style imitation or obfuscation.
- Training data is English-only; the model will not generalise to other languages.
- The closed-track dataset may not represent all writing domains or registers.
- Short texts provide fewer stylometric signals, reducing classification reliability.
- The model does not distinguish between style similarity and topic/content overlap.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameters were selected through experimentation. The model was compared against a Logistic Regression baseline during development, and the Gradient Boosting model was selected based on superior dev F1 (macro) score.
