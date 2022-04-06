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
import glob
from torchvision import transforms
import imageio

from models import *

# surface properties
length = 2
width = 2

resolution = (386, 516)

# training parameters
num_epochs = 20
lr = 1e-4
crop = 50

file_path = os.path.join('realSamples', '**', f'*.jpg')
paths = glob.glob(file_path, recursive=True)
numbers = [x[-6:-4] for x in paths]

images = [None]*len(numbers)
for idx, number in enumerate(numbers):
    if number[0] == '/':
        number = number[1]
    images[int(number)] = paths[idx]

convert_tensor = transforms.ToTensor()

samples = None
for image in images:
    try:
        imageGrayscale = imageio.imread(image)
    except:
        pass
    im = convert_tensor(Image.fromarray(imageGrayscale))[0].unsqueeze(0)
    if samples == None:
        samples = im
    else:
        samples = torch.cat((samples, im), dim=0)

'''h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[-r2, 0.0, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [0.0, -r2, h2],
                  [-r3, 0.0, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [0.0, -r3, h3]]'''

locationLights = [[ 0.1125, -2.5225,  1.2109],
        [ 2.5478, -0.1391,  1.1844],
        [ 0.0827,  2.5781,  1.1603],
        [-2.5279,  0.0720,  1.2012],
        [ 0.1713, -2.4175,  3.5454],
        [ 2.4348, -0.1907,  3.5272],
        [ 0.0684,  2.5286,  3.4532],
        [-2.7211,  0.1880,  3.4353],
        [ 0.1524, -2.1022,  5.5695],
        [ 2.0867,  0.1944,  5.5940],
        [-0.1161,  2.1110,  5.5742],
        [-2.0578,  0.1612,  5.5833]]

lights = torch.tensor(locationLights)

# cameraDistance = 8.0
camera = torch.tensor([[[[[-0.3831,  0.6247,  7.3251]]]]])

#paras = [(5, 0.1), (5, 0.5), (10, 0.1), (20, 0.1), (20, 0.01), (20, 0.001), (30, 0.01), (30, 0.001), (40, 0.01), (40, 0.001), (50, 0.01), (50, 0.001)]
paras = [[(1,0.1),
         (2,0.1),
         (2.5,0.1),
         (3,0.05),
         (4,0.05),
         (5,0.05),
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

            res = metric(pred[:,:,crop:-crop,crop:-crop], x[:,:,crop:-crop,crop:-crop])

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
        os.mkdir(os.path.join(path, f'{epoch}'))
        errs = update(model, trainloader, mse, optimizer)
        val_errs = evaluate(model, testloader, mse)

        surface_im = testset[0][0].unsqueeze(0).cuda()
        im = shader.forward(surface_im)
        pred = model(im)

        mse_surface = mse(surface_im, pred).item()

        diff.append(mse_surface)
        vals.append(mean(val_errs))
        trains.append(mean(errs))


        for L in range(12):
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(samples[L, crop:-crop, crop:-crop].cpu().detach().numpy())

            pred = model(samples.unsqueeze(0).to(next(model.parameters()).device))
            pred = shader.forward(pred).squeeze(0)
            plt.subplot(1, 2, 2)
            plt.imshow(pred[L, crop:-crop, crop:-crop].cpu().detach().numpy())

            plt.savefig(os.path.join(path, f'{epoch}', f'True-{L}.png'))
            plt.close()

            plt.imshow(im.squeeze(0)[L, crop:-crop, crop:-crop].cpu().detach().numpy())
            plt.savefig(os.path.join(path, f'{epoch}', f'Fake-{L}.png'))
            plt.close()

        '''for light in im_nr:
            im_target = im[0][light].unsqueeze(2).repeat(1, 1, 3)
            im_pred = shader.forward(pred)[0][light].unsqueeze(2).repeat(1, 1, 3)

            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(im_target.cpu().detach().numpy())

            plt.subplot(1, 2, 2)
            plt.imshow(im_pred.cpu().detach().numpy())

            plt.savefig(os.path.join(path, f'{epoch}-{light}.png'))
            plt.close()'''

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