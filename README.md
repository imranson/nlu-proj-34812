# COMP34812 NLU Coursework — Group 60

## Setup

1. Download the dataset zips and place them in the `downloads/` folder.
2. Extract `training_data.zip` and `trial_data.zip` to the project root so the structure matches:

**Do not push large files (datasets, model weights, etc.) to the repo.** The `downloads/`, `training_data/`, and `trial_data/` folders are gitignored — only the empty directories are tracked.

## Workflow

- **Create a branch for everything.** Do not commit directly to `main`.
- **Try not to merge to `main` unless everyone is around.** Coordinate merges so the whole team can review together.

## Code

- **Use Python scripts (`.py`) rather than notebooks for shared code.** Scripts are easier to review, merge, and version-control.
- Notebooks are fine for personal experimentation and exploration.

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
