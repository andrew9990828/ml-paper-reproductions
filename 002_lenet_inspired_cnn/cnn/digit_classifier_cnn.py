# =============================================================================
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last updated: 11/25/26
# File: cnn_pixel_shuffle.py
#
# MNIST Digit Classifier
#
# Mental Model
#
# 1. Load and batch the data 
#    Each example is a 28 x 28 grayscale image and a label from 0 to 9.
#    A batch gives us images [B, 1, 28, 28] and integer labels [B].
#    Set aside some of the training images for validation.
#   
# 2. Turn pixels into predictions
#    A convolution filter slides over the image and produces a feature map
#    showing where a pattern appears. The CNN uses those features to
#    produce 10 logits per image. A batch of logits has shape [B, 10].
#
# 3. Train on one batch at a time
#    Clear gradients, run a forward pass, and compute cross-entropy from
#    the logits and labels. Backpropagate and update the weights.
#    One pass through all the training batches is one epoch.
#
# 4. Check how well it generalizes
#    Run the model on validation images after each epoch. The largest
#    logit gives the predicted digit. Count how often it matches the label.
#    Use the separate test split for the final evaluation.
# =============================================================================
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.in_channels = in_channels  # 1 bc its grayscaled
        self.num_classes = num_classes  # nums 0-9 = 10 classes
        self.activation = nn.ReLU()     # our activation function
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # flatten before to get 32*7*7
        self.linear = nn.Linear(32*7*7, out_features=num_classes)
        # go from [1, 16] -> [16, 32] -> [32*7*7, 10]

    def forward(self, x:input):
        # [1, 1, 28, 28]
        x = self.conv1(x)
        # [1, 16, 28, 28]
        x = self.activation(x)
        # [1, 16, 28, 28]
        x = self.pool(x)
        # [1, 16, 14, 14]
        x = self.conv2(x)
        # [1, 32, 14, 14]
        x = self.activation(x)
        # [1, 32, 14, 14]
        pooled = self.pool(x)
        # [1, 32, 7, 7]
        logits = torch.flatten(pooled, start_dim=1, end_dim=-1)
        # [1, 1568]
        out = self.linear(logits)
        # [1, 10]
        return out


if __name__ == "__main__":
    data = torch.load("002_lenet_inspired_cnn/data/data_sets/training/train.pt", mmap=True, weights_only=True)
    print(type(data))
    image = data["images"][0]
    image = image.unsqueeze(0)
    x = image.unsqueeze(0).float()
    cnn = CNN()
    out = cnn(x)
    print(out)