import torch.nn as nn

import VAE.model as VAE


class CVAE(VAE.VariationalAutoEncoder):
    def __init__(self, shape, n_class, n_dims: int = 2, n_cond: int = 16):
        super(CVAE, self).__init__(shape, n_dims)
        self.ncond = n_cond
        if n_cond:
            self.labels = nn.Embedding(n_class, n_cond)

    def forward(self, x, y=None):
        if self.ncond and y is not None:
            y = self.labels(y)
        return super().forward(x, y)
