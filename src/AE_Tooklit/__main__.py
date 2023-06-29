import torchvision
import torch
import torch.nn as nn

from train import Trainer
from vis import compare
from AE.model import AutoEncoder
from VAE.model import VariationalAutoEncoder, var_loss
from CVAE.model import CVAE
from Classifier.model import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BCE = nn.BCELoss(reduction='sum')
CE = nn.CrossEntropyLoss(reduction='sum')


if __name__ == "__main__":
    print(device)

    shape = (1, 28, 28)
    n_class = 10

    n_dims = 32
    epochs = 10
    batch_size = 1024

    dataset = torchvision.datasets.FashionMNIST(
        "../../data/", transform=torchvision.transforms.ToTensor())
    testset = torchvision.datasets.FashionMNIST(
        "../../data/", False, transform=torchvision.transforms.ToTensor())

    net = Classifier(n_class).to(device)
    trainer = Trainer(net, dataset, testset,
                      lambda net, x, y: CE(net(x), y))
    trainer.train(epochs, batch_size)
    trainer.test(lambda net, x, y: (torch.max(net(x), 1)[1] == y).sum().item())

    net = AutoEncoder(shape, n_dims).to(device)
    trainer = Trainer(net, dataset, testset)
    trainer.train(epochs, batch_size)
    trainer.test(lambda net, x, y: BCE(net(x), x))
    compare(net, testset)

    net = VariationalAutoEncoder(shape, n_dims).to(device)
    trainer = Trainer(net, dataset, testset,
                      lambda net, x, y: var_loss(x, *net(x)))
    trainer.train(epochs, batch_size)
    trainer.test(lambda net, x, y: var_loss(x, *net(x)))
    compare(net, testset)

    net = CVAE(shape, n_class, n_dims, 64).to(device)
    trainer = Trainer(net, dataset, testset,
                      lambda net, x, y: var_loss(x, *net(x, y)))
    trainer.train(epochs, batch_size)
    trainer.test(lambda net, x, y: var_loss(x, *net(x, y)))
    compare(net, testset, True)
