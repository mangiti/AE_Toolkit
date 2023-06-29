import torch
import torch.nn as nn
import AE.model as AE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarEncoder(AE.Encoder):
    def __init__(self, n_dims: int = 2):
        super(VarEncoder, self).__init__(n_dims)
        self.log_var = nn.Sequential(
            nn.LazyLinear(4096), nn.ReLU(True),
            nn.Dropout(),
            nn.LazyLinear(4096), nn.ReLU(True),
            nn.LazyLinear(n_dims)
        )

    def forward(self, x, y=None):
        x = self.net(x)
        if y is not None:
            x = torch.cat([x, y], dim=1)
        return self.dense(x), self.log_var(x)


class VariationalAutoEncoder(AE.AutoEncoder):
    def __init__(self, shape, n_dims: int = 2):
        super(VariationalAutoEncoder, self).__init__(shape, n_dims)
        self.enc = VarEncoder(n_dims)

    def forward(self, x, y=None):
        mu, log_var = self.enc(x, y)
        eps = torch.randn_like(mu)
        sigma = 0.5 * torch.exp(log_var)
        z = mu + (sigma*eps)
        if y is not None:
            z = torch.cat([z, y], dim=1)
        return self.dec(z), mu, log_var


def var_loss(x, x_hat, mu, log_var):
    BCE_Loss = nn.BCELoss(reduction="sum")
    rec_loss = BCE_Loss(x_hat, x)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return rec_loss + kl_div
