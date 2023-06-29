import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, n_class):
        super(Classifier, self).__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(64, 5),
            nn.ReLU(True),  nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(128, 3),
            nn.ReLU(True),  nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(128, 3),
            nn.ReLU(True), nn.Dropout2d(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(n_class), nn.Softmax(1)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
