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


def train_NN(camera, lights, light_intensity, intensity, rough, diffuse, reflectance, shadow, x, y):
    resolution = (386, 516)

    # training parameters
    num_epochs = 40
    lr = 1e-4
    crop = 50

    file_path = os.path.join('realSamples', 'part5', f'*.jpg')
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

    model = ResidualNetwork()
    if torch.cuda.is_available():
        device = 'cuda'
        model.to(device)
    else:
        device = 'cpu'

    shader = FilamentShading(camera, lights, light_intensity, intensity, rough=rough, diffuse=diffuse, reflectance=reflectance, shadow=shadow, device=device,x=x,y=y)

    dataset = DummySet(resolution)
    ground_truth = torch.tensor(createSurface(resolution).tolist())

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

    #path = os.path.join('results', '20', 'Epoch-10000')
    #optimized_surface = torch.load(os.path.join(path, 'surface.pt'))

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
        comp_max = None
        for idx, err in enumerate(_forward(network, data, loss)):
            errs.append(err.item())

            opt.zero_grad()
            (err).backward()
            opt.step()

            if (idx % 2) == 0 and idx != 0:
                if not os.path.exists(os.path.join(path, 'video')):
                    os.mkdir(os.path.join(path, 'video'))

                im_gt = shader.forward(ground_truth.unsqueeze(0).to(device))
                pred_gt = model(im_gt).cpu().detach().numpy()[0,200,crop:-crop]

                x = np.linspace(0, len(pred_gt) - 1, len(pred_gt))
                plt.plot(x, ground_truth.cpu().detach().numpy()[200,crop:-crop], label='ground truth')
                plt.plot(x, pred_gt, label='prediction')
                plt.xlabel('x')
                plt.ylabel('height')
                plt.legend()
                plt.savefig(os.path.join(path, 'video', f'{epoch}-{idx}.png'))
                plt.close()

        return errs

    ############################################################################
    # training and evaluation
    ############################################################################

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    path = create_next_folder(os.path.join('results', 'trainNN'))

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

        comp_max = None

        for L in range(12):
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
        plt.close()

        torch.save(model.state_dict(), os.path.join(path, f'{epoch}', 'model.pt'))


        '''plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(comparism.squeeze(0)[crop:-crop, crop:-crop].cpu().detach().numpy())
        plt.title('camparism')
        plt.colorbar()
        #plt.clim(0, ground_truth.max())
        if comp_max == None:
            if abs(comparism.squeeze(0)[crop:-crop, crop:-crop].max()) > abs(comparism.squeeze(0)[crop:-crop, crop:-crop].min()):
                comp_max = comparism.squeeze(0)[crop:-crop, crop:-crop].max()
            else:
                comp_max = comparism.squeeze(0)[crop:-crop, crop:-crop].min()
        plt.clim(-abs(comp_max), abs(comp_max))

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth[crop:-crop, crop:-crop].cpu().detach().numpy())
        plt.title('ground truth')
        plt.colorbar()
        # plt.clim(0, 1.0)

        plt.subplot(1, 3, 3)
        plt.imshow(pred_gt.squeeze(0)[crop:-crop, crop:-crop].cpu().detach().numpy())
        plt.title('prediction')
        plt.colorbar()
        # plt.clim(0, 1.0)

        plt.savefig(os.path.join(path, f'{epoch}', f'compare.png'))

        plt.close()
'''
        # loading model
        '''
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        '''

        print(f'Epoch {epoch} AVG Mean {mean(errs):.6f} AVG Val Mean {mean(val_errs):.6f} MSE Surface {mse_surface}')