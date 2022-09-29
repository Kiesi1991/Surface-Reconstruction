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

optimized_parameters = getOptimizedParameters(os.path.join('results', 'optimization', '3', '3'))
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
        yield res

@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: torch.optim.Optimizer) -> list:
    network.train()
    errs = []
    for iter, err in tqdm(enumerate(_forward(network, data, loss))):
        errs.append(err.item())

        opt.zero_grad()
        (err).backward()
        opt.step()

        if (iter % 100) == 0:
            predicted_surface = model(real_samples.to(device).permute(0,4,2,3,1).squeeze(-1))  # (B,H,W)
            predicted_images = shader.forward((predicted_surface))  # (B,L,H,W)
            network.plotImageComparism(real_samples, predicted_images, path)
    return errs

############################################################################
# training and evaluation
############################################################################

mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

os.mkdir(os.path.join(path, f'{iter}'))
errs = update(model, trainloader, mse, optimizer)

''' for L in range(12):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(samples[L, crop:-crop, crop:-crop].cpu().detach().numpy())
    plt.clim(0, 1.0)

    pred = model(samples.unsqueeze(0).to(next(model.parameters()).device))
    pred = shader.forward(pred).squeeze(0)
    plt.subplot(1, 2, 2)
    plt.imshow(pred[L, crop:-crop, crop:-crop].cpu().detach().numpy())
    plt.clim(0, 1.0)

    plt.savefig(os.path.join(path, f'{epoch}', f'True-{L}.png'))
    plt.close()

    plt.imshow(im.squeeze(0)[L, crop:-crop, crop:-crop].cpu().detach().numpy())
    plt.clim(0, 1.0)
    plt.savefig(os.path.join(path, f'{epoch}', f'Fake-{L}.png'))
    plt.close()

    p = cv2.cvtColor(pred[L, crop:-crop, crop:-crop].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
    t = cv2.cvtColor(samples[L, crop:-crop, crop:-crop].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(t)
    plt.clim(0, 1.0)

    plt.subplot(1, 2, 2)
    plt.imshow(p)
    plt.clim(0, 1.0)

    plt.savefig(os.path.join(path, f'{epoch}', f'TrueRGB-{L}.png'))
    plt.close()

im_gt = shader.forward(ground_truth.unsqueeze(0).to(device))
pred_gt = model(im_gt)
#comparism = pred_gt - ground_truth.unsqueeze(0).to(device)

true_mse_surface = torch.log(mse(pred_gt, ground_truth.unsqueeze(0).to(device)) + 1).item()
diff_surface.append(true_mse_surface)

x = np.linspace(0, len(diff_surface) - 1, len(diff_surface))
plt.plot(x, diff_surface, label='difference')
plt.xlabel('epoch')
plt.ylabel('Error')
plt.legend()
plt.savefig(os.path.join(path, f'difference-surface.png'))
plt.close()

x = np.linspace(0, len(diff) - 1, len(diff))
plt.plot(x, diff, label='difference')
plt.plot(x, vals, label='validation')
plt.plot(x, trains, label='training')
plt.xlabel('epoch')
plt.ylabel('Error')
plt.legend()
plt.savefig(os.path.join(path, f'training-errors.png'))
plt.close()'''

torch.save(model.state_dict(), os.path.join(path, f'{iter}', 'model.pt'))

print(f'Iterations {iter} AVG Mean {mean(errs):.6f}')