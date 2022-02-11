import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import *
import numpy as np
from shaders import PhongShading
from dataset import DummySet
from PIL import Image
from statistics import mean
import matplotlib.pyplot as plt

from models import zPrediction

# surface properties
length = 2
width = 2

resolution = (512, 512)

# training parameters
num_epochs = 20
lr = 1e-3

surface = createSurface(resolution)

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[-r1, 0.0, h1], [r1, 0.0, h1],
                  [0.0, r1, h1],  [0.0, -r1, h1],
                  [-r2, 0.0, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [0.0, -r2, h2],
                  [-r3, 0.0, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [0.0, -r3, h3]]

cameraDistance = 8.0

dataset = DummySet(resolution)

model = zPrediction()
if torch.cuda.is_available():
    device = 'cuda'
    model.to(device)
else:
    device = 'cpu'

shader = PhongShading(camera=[0, 0, cameraDistance], lights=locationLights, length=length, width=width, device=device)
#I = shader.forward(dataset.data[0])

#for idx, i in enumerate(I):
#    im = Image.fromarray(np.uint8(i*255))
#    im.show()

n_samples = len(dataset)
# Shuffle integers from 0 n_samples to get shuffled sample indices
shuffled_indices = np.random.permutation(n_samples)
testset_inds = shuffled_indices[:n_samples//10]
trainingset_inds = shuffled_indices[n_samples//10:]

from torch.utils.data import Subset
# Create PyTorch subsets from our subset-indices
testset = Subset(dataset, indices=testset_inds)
trainingset = Subset(dataset, indices=trainingset_inds)

testloader = DataLoader(testset, batch_size=1, shuffle=False)
trainloader = DataLoader(trainingset, batch_size=4, shuffle=True)

############################################################################
# Update and evaluate network
############################################################################

def _forward(network: nn.Module, data: DataLoader, metric: callable):
    device = next(network.parameters()).device

    for j, (surface, idx) in enumerate(data):
        res = 0
        surface = surface.to(device)
        x = shader.forward(surface)

        pred = model(x)
        pred = shader.forward((pred))

        res = metric(pred, x)

        yield res

@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> list:
    network.eval()

    results = _forward(network, data, metric)
    return [res.item() for res in results]


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: torch.optim.Optimizer) -> list:

    network.train()

    errs = []
    for idx, err in enumerate(_forward(network, data, loss)):
        errs.append(err.item())

        opt.zero_grad()
        err.backward()
        opt.step()
    return errs

############################################################################
# training and evaluation
############################################################################

mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

path = os.path.join('results')

folder = 0
while True:
    if not os.path.exists(os.path.join(path)):
        os.mkdir(os.path.join(path))
    if not os.path.exists(os.path.join(path, f'{folder}')):
        os.mkdir(os.path.join(path, f'{folder}'))
        path = os.path.join(path, f'{folder}')
        break
    else:
        folder += 1

im_nr = 5
for epoch in range(num_epochs):
    errs = update(model, trainloader, mse, optimizer)
    val_errs = evaluate(model, testloader, mse)

    im = shader.forward(testset[0][0].unsqueeze(0).cuda())
    pred = model(im)

    im = im[0][im_nr].unsqueeze(2).repeat(1, 1, 3)
    im_pred = shader.forward(pred)[0][im_nr].unsqueeze(2).repeat(1, 1, 3)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(im.cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(im_pred.cpu().detach().numpy())

    plt.savefig(os.path.join(path, f'{epoch}.png'))

    print(f'Epoch {epoch} AVG Mean {mean(errs):.6f} AVG Val Mean {mean(val_errs):.6f}')

print('TheEnd')