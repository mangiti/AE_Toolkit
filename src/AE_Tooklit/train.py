import torch
import torch.nn as nn
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BCE = nn.BCELoss(reduction='sum')


class Trainer():
    def __init__(self, net: nn.Module, data, testData,  update_fn=None):
        self.net = net
        self.trainData = data
        self.testData = testData
        self.update_fn = update_fn if update_fn is not None else \
            lambda net, x, y: BCE(net(x), x)

    def train(self, epochs: int = 10, batch_size: int = 128):

        opt = torch.optim.Adam(self.net.parameters())
        train = torch.utils.data.DataLoader(
            self.trainData, batch_size=batch_size, shuffle=True)

        self.net.train()
        for epoch in range(epochs):
            tbar = tqdm.tqdm(train, ncols=90)
            for x, y in tbar:
                x = x.to(device)
                y = y.to(device)

                loss = self.update_fn(self.net, x, y)

                loss.backward()
                opt.step()
                opt.zero_grad()

                tbar.set_description(f'Epoch {epoch} | Loss {loss.item():.5E}')

    def test(self, test_fn):
        test = torch.utils.data.DataLoader(
            self.testData, batch_size=10, shuffle=True)
        self.net.eval()
        with torch.no_grad():
            sum = 0
            total = 0
            for x, y in test:
                x, y = x.to(device), y.to(device)
                total += y.size(0)
                sum += test_fn(self.net, x, y)

        print(f"Evaluated avg loss: {100*sum/total:.2F}")
