import torch

DEFAULTS = {
    # data
    "train_csv": "training_data/AV/train.csv",
    "dev_csv": "training_data/AV/dev.csv",

    # model
    "model_name": "microsoft/deberta-v3-large",
    "max_length": 512,

    # contrastive loss
    "margin": -0.5,

    # training
    "epochs": 3,
    "batch_size": 8,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "fp16": True,

    # runtime
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0,
    "log_steps": 100,
    "output_dir": "checkpoints/av_siamese",
}
