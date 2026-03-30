# COMP34812 NLU Coursework — Group 60

## Authorship Verification (AV) Track Description
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
**Approach:** Siamese approach with BERT variant and contrastive loss. View model card for more information.

**Key features:** Cosine similarity, contrastive loss
#### Running the Demo (Inference)
1. Navigate to `solution_C/`
2. Open `demo_solution_C.ipynb`
3. Set `MODEL_PATH` and `TEST_CSV` in at the top
4. Download trained model from https://livemanchesterac-my.sharepoint.com/:u:/g/personal/chun_tham_student_manchester_ac_uk/IQBlmeb7lWvaRqApkmkxD1VPATbhm88zhEfKLNnH1m_EY4I?e=AyjL9H
5. Run all cells — predictions will be saved to the specified output file
#### Training from Scratch in Terminal
1. Navigate to `solution_C/`
2. Run `python train_code_C.py --max_len 512 --margin 0.5 --dist_lower 0 --train_csv ../training_data/AV/train.csv --dev_csv ../training_data/AV/dev.csv --model_name microsoft/deberta-v3-xsmall --epochs 300 --batch_size 16 --lr 1e-5 --save_model --output_dir . --save_model`
#### Running Prediction Code (Same as Demo Code)
1. Navigate to `solution_C/`
2. Download trained model from https://livemanchesterac-my.sharepoint.com/:u:/g/personal/chun_tham_student_manchester_ac_uk/IQBlmeb7lWvaRqApkmkxD1VPATbhm88zhEfKLNnH1m_EY4I?e=AyjL9H
3. Run `python pred_code_C.py --test_csv test_data/AV/test.csv --model_name microsoft/deberta-v3-xsmall --model_path deberta-v3-xsmall_ep300_bs16_lr1e-05_ml512_m0.5_dl0_model.pt --batch_size 64 --max_len 512 --margin 0.5 --dist_lower 0 --output_path Group_60_C.csv`
#### Other Code
1. `config.py` contains default values for `train_code_C.py` and `pred_code_C.py`.

## Attribution
All training, development, and test data were provided as part of the COMP34812 AV shared task (closed track). No external datasets were used in accordance with the closed-track rules.

For solution C, below are the relevant sources for the BERT variant used:
Base model repo: https://huggingface.co/microsoft/deberta-v3-xsmall
Base model paper: https://doi.org/10.48550/arXiv.2111.09543
## Model Artefacts
All model files for Solution A are under 10 MB total (~1.6 MB) and are included directly in the submission. No cloud-hosted models are required.

For Solution C, download trained model from https://livemanchesterac-my.sharepoint.com/:u:/g/personal/chun_tham_student_manchester_ac_uk/IQBlmeb7lWvaRqApkmkxD1VPATbhm88zhEfKLNnH1m_EY4I?e=AyjL9H (~283 MB total).