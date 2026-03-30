---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/imranson/nlu-proj-34812

---

# Model Card for q12628ct-u93792gc-m13447tk-AV

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

- **Repository:** https://huggingface.co/microsoft/deberta-v3-xsmall
- **Paper or documentation:** https://doi.org/10.48550/arXiv.2111.09543

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

27K pairs of texts provided with the assignment, with 0/1 labels.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 1e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: None
      - num_epochs: 300

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 2 days 7 hours
      - duration per training epoch: 11 minutes
      - model size: 283MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Development set and trial dataset provided with the assignment, amounting to 6K and 50 pairs respectively.

#### Metrics

<!-- These are the evaluation metrics being used. -->


    Dev set:
      - Precision: 0.7653
      - Recall: 0.8920
      - Macro F1-score: 0.8033
      - Accuracy: 0.8054
      - AUC: 0.8037
    
    Trial set:
      - Precision: 0.8889
      - Recall: 0.9600
      - Macro F1-score: 0.9199
      - Accuracy: 0.9200
      - AUC: 0.9200

### Results

The model obtained a macro F1-score of 0.8033 and AUC of 0.8037. Macro F1 equally weighs the F1 score from each class. The classes in the dev set are roughly balanced. 

## Technical Specifications

### Hardware


    For evaluation with the default demo code configurations,
      - RAM: at least 4GB,
      - VRAM: at least 4GB,
      - Storage: at least 300MB,
      - GPU: T5
    
    Batch size can be reduced to use less RAM and VRAM.

### Software


    Tested on:
      - Transformers 5.0.0
      - Pytorch 2.10.0+cu128
      - Pandas 2.2.2

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any single piece of text longer than 512 tokens (tokenized by the same model) will be truncated by the model. Shorter sequences will be padded, which may affect accuracy.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by manual experimentation with different values. With more time, better generalization is probably achievable via tuning and additional hyperparameters e.g. dropout, weight decay etc.
