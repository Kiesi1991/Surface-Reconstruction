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



resolution = (386, 516)

# training parameters
num_epochs = 40
lr = 1e-4
crop = 50

real_samples = getRealSamples('realSamples1')

model = ResidualNetwork()
if torch.cuda.is_available():
    device = 'cuda'
    model.to(device)
else:
    device = 'cpu'

optimized_parameters = getOptimizedParameters(os.path.join('results', 'optimization', '3', '3'))
shader = FilamentShading(optimized_parameters)
dataset = DummySet(resolution)

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
    for synthetic_surface, idx in data:
        synthetic_images = shader.forward(synthetic_surface.to(device)) # (B,L,H,W)
        predicted_surface = model(synthetic_images) # (B,H,W)
        predicted_images = shader.forward((predicted_surface)) # (B,L,H,W)
        res = metric(predicted_images[..., crop:-crop, crop:-crop],
                     synthetic_images[..., crop:-crop, crop:-crop])
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
    for err in _forward(network, data, loss):
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

path = createNextFolder(os.path.join('results', 'trainNN'))

diff = []
diff_surface = []
vals = []
trains = []
for epoch in range(num_epochs):
    os.mkdir(os.path.join(path, f'{epoch}'))
    errs = update(model, trainloader, mse, optimizer)
    val_errs = evaluate(model, testloader, mse)

    surface_im = testset[0][0].unsqueeze(0).cuda() # 1,H,W
    im = shader.forward(surface_im) # 1,L,H,W
    pred = model(im) # 1,H,W

    mse_surface = mse(surface_im, pred).item()

    diff.append(mse_surface)
    vals.append(mean(val_errs))
    trains.append(mean(errs))

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

    torch.save(model.state_dict(), os.path.join(path, f'{epoch}', 'model.pt'))

    print(f'Epoch {epoch} AVG Mean {mean(errs):.6f} AVG Val Mean {mean(val_errs):.6f} MSE Surface {mse_surface}')