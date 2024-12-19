project_name = "ImageClassification"

model = {
    "name": "ResNet18",
    "num_layers": 18
}

dataset = {
    "name": "CIFAR100",
    "batch_size": 1024,
    "train_split": 0.7,
    "val_split": 0.3,
    "augumentations": True
}

max_epochs = 30

optimizer = {
    "name": "Adam",
    "weight_decay": None
}

# lr_scheduler = "default"

lr_scheduler = {
    "name": "OneCycleLR",
    "max_lr": 0.01,
    "anneal_strategy": "cos"
}

wandb_offline = False
fast_dev_run = False
