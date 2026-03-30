# COMP34812 NLU Coursework — Group 60

## Authorship Verification (AV) Track

### Task Description
Given two text sequences, determine whether both were written by the same author (binary classification: 0 = different author, 1 = same author).
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
**Approach:** Siamese approach with BERT variant and contrastive loss
## Data Attribution
All training, development, and test data were provided as part of the COMP34812 AV shared task (closed track). No external datasets were used in accordance with the closed-track rules.
## Model Artefacts
All model files for Solution A are under 10 MB total (~1.6 MB) and are included directly in the submission. No cloud-hosted models are required.
