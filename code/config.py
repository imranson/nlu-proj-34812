import torch

DEFAULTS = {
    # data
    "train_csv": "/scratch/u93792gc/nlu_project/training_data/AV/train.csv",
    "dev_csv": "/scratch/u93792gc/nlu_project/training_data/AV/dev.csv",
    "embedding_path": "/scratch/u93792gc/nlu_project/training_data/crawl-300d-2M-subword/crawl-300d-2M-subword.vec",
    "output_dir": "output",
    "max_length": 150,  # 150 gave best performance tested 75, 100, 110, 120, ..., 190, 200
    "data_augmentation": False, # False best
    "lowercase": True, # True best

    # model
    "embedding_dim": 300,
    "hidden_dim": 112,  # tested 16, 32, 64, 96, 112, 120, 128, 192, 256, 512
    "num_layers": 1,  # Tested 1-4 layers and 1 performed best
    "lstm_dropout": 0.0,  # only useful for num_layers > 1
    "freeze_embeddings": True, # True best
    "pos_weight": 0.9, # tested 0.5, 0.75, 0.9, 1.0, 1.25, 1.5

    # training
    "epochs": 40, # no more is necessary
    "batch_size": 8,  # 8 gave best perofmance, tested 4, 8, 16, 32, 48, 64
    "lr": 5e-3,  # 5e-3 best, tested 5e-4, 1e-4, 1e-3
    "weight_decay": 0.005,  # tested 0, 0.001, 0.005, 0.01, 0.05, 0.1
    "max_grad_norm": 1.0,  # tested 0.0, 0.5, 2.0, 5.0
    "num_workers": 8,
    "log_steps": 200, 
    "dropout": 0.005, # tried 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75
    "seed": 42,
    "loss": "bce", # bce best, tested bce with logits

    # runtime
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "test_csv": "/scratch/u93792gc/nlu_project/training_data/AV/dev.csv", #temporary until test set is available
    "checkpoint": "output/best_model.pt",
}