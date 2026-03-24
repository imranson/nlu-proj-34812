import torch

DEFAULTS = {
    # data
    "train_csv": "training_data/AV/train.csv",
    "dev_csv": "training_data/AV/dev.csv",

    # model
    "model_name": "microsoft/deberta-v3-xsmall",
    "max_length": 50,

    # contrastive loss
    "margin": 1,
    "dist_lower": 0,

    # training
    "epochs": 100,
    "batch_size": 64,
    "lr": 2e-5,
    # "weight_decay": 0.01,
    # "warmup_ratio": 0.1,
    # "max_grad_norm": 1.0,
    # "fp16": True,

    # runtime
    "device": torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu",
    # "num_workers": 0,
    # "log_steps": 100,
    "output_dir": "checkpoints/av_siamese",
}
