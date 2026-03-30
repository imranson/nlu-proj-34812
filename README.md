# COMP34812 Coursework — Group 60
## Authorship Verification (AV) Track

### Task Description
Given two text sequences, determine whether both were written by the same author (binary classification: 0 = different author, 1 = same author).

### Repository Structure
```
coursework/
├── README.md
├── COMP34812_modelcard_template.md
├── Model-Card-Creation.ipynb
├── training_data/
│   └── AV/
│       ├── train.csv
│       └── dev.csv
├── test_data/
│   └── AV/
│       └── test.csv
├── trial_data/
│   ├── AV_trial.csv
│   ├── ED_trial.csv
│   └── NLI_trial.csv
├── solution_A/
│   ├── train_solution_A.py
│   ├── train_solution_A.ipynb
│   ├── demo_solution_A.ipynb
│   ├── model_card_A.md
│   ├── 60_A.csv
│   └── models/
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── char_vectorizer.pkl
│       ├── word_vectorizer.pkl
│       └── feature_names.pkl
└── solution_C/
    └── [To be added upon merge]
```

### Solution A: Traditional Machine Learning (Category A)
**Approach:** Stylometric feature engineering + Gradient Boosting classifier

**Key features:** Punctuation usage rates, vocabulary richness (TTR, hapax, Yule's K), function word frequencies, text structure statistics, pairwise stylometric distances (NCD, Burrows' Delta, Keselj dissimilarity), TF-IDF cosine similarities.

#### Running the Demo (Inference)
1. Navigate to `solution_A/`
2. Open `demo_solution_A.ipynb`
3. Set `MODEL_DIR` and `INPUT_FILE` in the configuration cell
4. Run all cells — predictions will be saved to the specified output file

#### Training from Scratch
1. Navigate to `solution_A/`
2. Open `train_solution_A.ipynb`
3. Set `TRAIN_FILE`, `DEV_FILE`, and `MODEL_DIR` in the configuration cell
4. Run all cells — trained model artefacts will be saved to `MODEL_DIR`

### Solution C: Transformer-based (Category C)
To be completed upon merge of Solution C branch.

### Data Attribution
All training, development, and test data were provided as part of the COMP34812 AV shared task (closed track). No external datasets were used in accordance with the closed-track rules.

### Model Artefacts
All model files for Solution A are under 10 MB total (~1.6 MB) and are included directly in the submission. No cloud-hosted models are required.
