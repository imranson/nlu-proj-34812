---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/imranson/nlu-proj-34812

---

# Model Card for q12628ct-username2-AV

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to detect whether two pieces of text were written by the same author.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is a Siamese model that implements contrastive loss. A Siamese model is when different inputs are fed into the exact same model. Contrastive loss is a loss function that calculates loss based on the distance between 2 embeddings. The contrastive loss function takes a margin parameter. Margin is a chosen threshold that causes the contrastive loss function to penalize embedding pairs that are incorrectly distanced from each other. For example, given a document embedding pair with texts from different authors, the loss function penalizes the embedding pair if they have a embedding distance below the chosen margin threshold. The distance measure used in this case is (1 - cosine similarity).

The document embeddings are produced using a BERT model variant called microsoft/deberta-v3-xsmall on Hugging Face. For each sample, the 2 texts are fed into the model separately. The 384-dimension CLS tokens from the 2 outputs are used to calculate distance and perform classification. This model was fine-tuned on 27K pairs of texts provided with this assignment.

- **Developed by:** Tham Chun Shen
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** microsoft/deberta-v3-xsmall

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://doi.org/10.48550/arXiv.2111.09543

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

27K pairs of texts provided with the assignment, with 0/1 labels.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: 42
      - num_epochs: 10

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 5 hours
      - duration per training epoch: 30 minutes
      - model size: 300MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 2K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 67% and an accuracy of 70%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Transformers 4.18.0
      - Pytorch 1.11.0+cu113

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any single piece of text longer than 512 tokens (tokenized by the same model) will be truncated by the model. Shorter sequences will be padded, which may affect accuracy.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by manual experimentation with different values. With more time, better generalization is probably achievable via additional hyperparameters e.g. dropout, weight decay etc.
