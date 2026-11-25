import torch
import torchvision
from torch.utils.data import random_split

from pathlib import Path

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

data_dir = Path(__file__).resolve().parents[1] / "data"
data_dir = Path(__file__).resolve().parents[1] / "data"
train_dir = data_dir / "data_sets" / "training"
val_dir = data_dir / "data_sets" / "validation"

digits = MNIST(root=data_dir, train=True, download=True, transform=ToTensor())

# Splits I chose.
train_split, validation_split = 0.9, 0.1

# Split the data.
train_data, validation_data = random_split(digits, (train_split, validation_split))

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "images": digits.data[train_data.indices],
        "labels": digits.targets[train_data.indices],
    },
    train_dir / "train.pt",
)

torch.save(
    {
        "images": digits.data[validation_data.indices],
        "labels": digits.targets[validation_data.indices],
    },
    val_dir / "validation.pt",
)