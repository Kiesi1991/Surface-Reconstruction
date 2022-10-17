from torch.utils.data import DataLoader
from torch.utils.data import Subset

from utils import *
import numpy as np
from dataset import DummySet
from PIL import Image
from statistics import mean
import matplotlib.pyplot as plt
from shaders import FilamentShading
import glob
from torchvision import transforms
import imageio
import cv2
from models import *
from tqdm import tqdm



resolution = (386, 516)

# training parameters
num_iter = 10000
batch_size = 4
lr = 1e-4
crop = 50
layers=6

real_samples = getRealSamples('realSamples1')
path = createNextFolder(os.path.join('results', 'trainNN'))

model = ResidualNetwork(layers=layers, crop=crop)
if torch.cuda.is_available():
    device = 'cuda'
    model.to(device)
else:
    device = 'cpu'

try:
    optimized_parameters = getOptimizedParameters(os.path.join('results', 'optimization', '1', '0'))
except:
    raise ('Please enter a folder path including optimized parameters')
shader = FilamentShading(optimized_parameters)
dataset = DummySet(resolution, amount_data=num_iter)

trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

############################################################################
# Update network
############################################################################

def _forward(network: nn.Module, data: DataLoader, metric: callable):
    device = next(network.parameters()).device
    for synthetic_surface, idx in data:
        synthetic_images = shader.forward(synthetic_surface.to(device)) # (B,L,H,W)
        predicted_surface = model(synthetic_images) # (B,H,W)
        predicted_images = shader.forward((predicted_surface)) # (B,L,H,W)
        res = metric(predicted_images[..., crop:-crop, crop:-crop],
                     synthetic_images[..., crop:-crop, crop:-crop])
        mse_surface = metric(synthetic_surface.to(device),
                             predicted_surface.to(device))
        yield res, mse_surface

@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: torch.optim.Optimizer) -> list:
    network.train()
    errs = []
    for iter, (err, mse_surface) in tqdm(enumerate(_forward(network, data, loss))):
        errs.append(err.item())

        opt.zero_grad()
        (err).backward()
        opt.step()

        if (iter % 100) == 0:
            predicted_surface = model(real_samples.to(device).permute(0,4,2,3,1).squeeze(-1))  # (B,H,W)
            predicted_images = shader.forward((predicted_surface))  # (B,L,H,W)
            network.plotImageComparism(real_samples, predicted_images, path)
            network.plotProfileDiagrams(shader.optimized_surface, predicted_surface, path)
            network.plotErrorDiagram(errs, path)
            torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    return errs

############################################################################
# training and evaluation
############################################################################

mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

errs = update(model, trainloader, mse, optimizer)