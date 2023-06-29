import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compare(net, data, labels=False, limit=10):
    testData = torch.utils.data.DataLoader(
        data, batch_size=10, shuffle=True)

    net.eval()
    with torch.no_grad():
        for x, y in testData:
            x = x.to(device)
            if labels:
                y = y.to(device)
                recon = net(x, y)
            else:
                recon = net(x)

            if isinstance(recon, tuple):
                recon = recon[0]

            x = np.hstack([img for img in x.cpu().numpy()])
            recon = np.hstack([img for img in recon.cpu().numpy()])

            imgs = np.moveaxis(np.dstack([x, recon]), 0, 2)
            plt.imshow(imgs)
            plt.show()
            if limit == 0:
                break
            else:
                limit -= 1


def genSquare(net, n_class, device='cpu'):
    with torch.no_grad():
        imgStack = []
        for i in range(n_class):
            imgStack.append(
                np.hstack([img for img in net.gen(np.arange(1, n_class+1), device)]))
        imgs = np.dstack(imgStack)
        imgs = np.moveaxis(imgs, 0, 2)
        plt.imshow(imgs)
        plt.show()
