import torch

DEFAULTS = {
    # data
    "train_csv": "training_data/AV/train.csv",
    "dev_csv": "training_data/AV/dev.csv",
    "embedding_path": "training_data/crawl-300d-2M-subword/crawl-300d-2M-subword.vec",
    "output_dir": "output",
    "max_length": 100,  # will optimise
    "data_augmentation": False, # will optimise
    "lowercase": False, # will optimise

    # model
    "embedding_dim": 300,
    "hidden_dim": 128,  # will optimise
    "num_layers": 1,  # will optimise
    "lstm_dropout": 0.0,  # will optimise
    "freeze_embeddings": False, # currently optimising

    # training
    "epochs": 10,  # will optimise
    "batch_size": 32,  # will optimise
    "lr": 1e-3,  # will optimise
    "weight_decay": 0.01,  # will optimise
    "max_grad_norm": 1.0,  # will optimise
    "num_workers": 0,
    "log_steps": 200, 
    "dropout": 0.3, # will optimise
    "seed": 42,
    "loss": "bce", # will optimise by trying BCEWithLogitsLoss
    "patience": 4, 

    # runtime
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}