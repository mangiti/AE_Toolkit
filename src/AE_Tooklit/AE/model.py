import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,  n_dims: int = 2):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(64, 5),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.LazyConv2d(64, 5),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(128, 3),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.LazyConv2d(256, 3),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.LazyLinear(2048), nn.ReLU(True)
        )

        self.dense = nn.Sequential(
            nn.LazyLinear(2048), nn.ReLU(True),
            nn.LazyLinear(2048), nn.ReLU(True),
            nn.LazyLinear(n_dims)
        )

    def forward(self, x):
        x = self.net(x)
        return self.dense(x)


class Decoder(nn.Module):
    def __init__(self, shape):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.net = nn.Sequential(
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(1024), nn.ReLU(True),
            nn.LazyLinear(c*w*h), nn.Sigmoid()
        )

    def forward(self, x):
        c, w, h = self.shape
        return self.net(x).view(-1, c, w, h)


class AutoEncoder(nn.Module):
    def __init__(self, shape, n_dims: int = 2):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(n_dims)
        self.dec = Decoder(shape)

    def forward(self, x: torch.Tensor):
        x = self.enc(x)
        x = self.dec(x)
        return x
