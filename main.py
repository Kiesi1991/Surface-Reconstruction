import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import *
import numpy as np
from shaders import PhongShading
from dataset import DummySet
from PIL import Image

from models import zPrediction

# surface properties
length = 4
width = 2

resolution = (512, 1028)

# training parameters
num_epochs = 2
lr = 1e-4

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

shader = PhongShading(camera=[0, 0, cameraDistance], lights=locationLights, length=length, width=width)
I = shader.forward(dataset.data[0])

for idx, i in enumerate(I):
    im = Image.fromarray(np.uint8(i*255))
    im.show()

n_samples = len(dataset)
# Shuffle integers from 0 n_samples to get shuffled sample indices
shuffled_indices = np.random.permutation(n_samples)
testset_inds = shuffled_indices[:n_samples//10]
trainingset_inds = shuffled_indices[n_samples//10:]

from torch.utils.data import Subset
# Create PyTorch subsets from our subset-indices
testset = Subset(dataset, indices=testset_inds)
trainingset = Subset(dataset, indices=trainingset_inds)

############################################################################
# Update and evaluate network
############################################################################

def _forward(network: nn.Module, data: DataLoader, metric: callable, i=None, evaluate=False):
    device = next(network.parameters()).device

    for j, (surface, idx) in enumerate(data):
        res = 0
        surface = surface.to(dtype=torch.float32)
        x = shader.forward(surface)
        y = y.to(dtype=torch.float32)

        yield res

@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable, i) -> list:
    network.eval()

    results = _forward(network, data, metric, i=i, evaluate=True)
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

model = zPrediction()
mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(num_epochs):
    errs = update(model, trainingset, mse, optimizer)


print('TheEnd')