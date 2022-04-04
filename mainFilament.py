import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from utils import *
import numpy as np
from shaders import PhongShading
from dataset import DummySet
from PIL import Image
from statistics import mean
import matplotlib.pyplot as plt
from shaders import FilamentShading

from models import *

# surface properties
length = 2
width = 2

resolution = (386, 516)

# training parameters
num_epochs = 20
lr = 1e-4

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[-r2, 0.0, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [0.0, -r2, h2],
                  [-r3, 0.0, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [0.0, -r3, h3]]

lights = torch.tensor(locationLights)

cameraDistance = 8.0
camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])

#paras = [(5, 0.1), (5, 0.5), (10, 0.1), (20, 0.1), (20, 0.01), (20, 0.001), (30, 0.01), (30, 0.001), (40, 0.01), (40, 0.001), (50, 0.01), (50, 0.001)]
paras = [[(5,0.05),
         (10,0.05),
         (20,0.05),
         (30,0.05),
         (40,0.05),
         (50,0.05)]]

for j, paras in enumerate(paras):
    model = ResidualNetwork()
    if torch.cuda.is_available():
        device = 'cuda'
        model.to(device)
    else:
        device = 'cpu'

    shader = FilamentShading(camera, lights, device=device)

    dataset = DummySet(resolution, para=paras)

    n_samples = len(dataset)
    # Shuffle integers from 0 n_samples to get shuffled sample indices
    shuffled_indices = np.random.permutation(n_samples)
    testset_inds = shuffled_indices[:n_samples//10]
    trainingset_inds = shuffled_indices[n_samples//10:]

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
            surface = surface.to(device)
            x = shader.forward(surface)

            pred = model(x)
            pred = shader.forward((pred))

            res = metric(pred[:,:,40:-40,40:-40], x[:,:,40:-40,40:-40])

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
            (err).backward()
            opt.step()
        return errs

    ############################################################################
    # training and evaluation
    ############################################################################

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    path = os.path.join('results')

    folder = 0
    while True:
        if not os.path.exists(os.path.join(path)):
            os.mkdir(os.path.join(path))
        if not os.path.exists(os.path.join(path, f'{folder}')):#, f'sigma={sigma},p={p}')):
            os.mkdir(os.path.join(path, f'{folder}'))#, f'sigma={sigma},p={p}'))
            path = os.path.join(path, f'{folder}')#, f'sigma={sigma},p={p}')
            break
        else:
            folder += 1

    im_nr = [1, 5]
    diff = []
    vals = []
    trains = []
    for epoch in range(num_epochs):
        errs = update(model, trainloader, mse, optimizer)
        val_errs = evaluate(model, testloader, mse)

        surface_im = testset[0][0].unsqueeze(0).cuda()
        im = shader.forward(surface_im)
        pred = model(im)

        mse_surface = mse(surface_im, pred).item()

        diff.append(mse_surface)
        vals.append(mean(val_errs))
        trains.append(mean(errs))

        for light in im_nr:
            im_target = im[0][light].unsqueeze(2).repeat(1, 1, 3)
            im_pred = shader.forward(pred)[0][light].unsqueeze(2).repeat(1, 1, 3)

            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(im_target.cpu().detach().numpy())

            plt.subplot(1, 2, 2)
            plt.imshow(im_pred.cpu().detach().numpy())

            plt.savefig(os.path.join(path, f'{epoch}-{light}.png'))
            plt.close()

        print(f'Epoch {epoch} AVG Mean {mean(errs):.6f} AVG Val Mean {mean(val_errs):.6f} MSE Surface {mse_surface}')

    x = np.linspace(0, len(diff)-1, len(diff))
    plt.plot(x, diff, label='difference')
    plt.plot(x, vals, label='validation')
    plt.plot(x, trains, label='training')
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(os.path.join(path, f'{epoch}.png'))
    plt.close()

print('TheEnd')