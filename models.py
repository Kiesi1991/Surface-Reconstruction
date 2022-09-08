from torch import nn
from torch.nn.parameter import Parameter
import torchvision
import torch.nn.functional as F
from utils import *
import os
import matplotlib.pyplot as plt
import cv2
import statistics

###############################################
#                ZPrediction                  #
###############################################

class zPrediction(nn.Module):
    def __init__(self):
        super(zPrediction, self).__init__()
        self.conv1 = nn.Conv2d(12, 96, kernel_size=3, padding=3 // 2)
        self.conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv2d(48, 24, kernel_size=3, padding=3 // 2)
        self.conv4 = nn.Conv2d(24, 12, kernel_size=3, padding=3 // 2)
        self.conv5 = nn.Conv2d(12, 1, kernel_size=3, padding=3 // 2)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, x):
        x = x.float()
        x = self.selu(self.conv1(x))
        x = self.selu(self.conv2(x))
        x = self.selu(self.conv3(x))
        x = self.selu(self.conv4(x))
        x = self.conv5(x)

        return x.squeeze(1)

###############################################
#               ResidualNetwork               #
###############################################

class ResidualNetwork(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        self.begin = nn.Conv2d(12, 12, kernel_size=1)
        self.res_blocks = nn.ModuleList([ResBlock(12) for i in range(layers)])
        self.head = nn.Conv2d(12, 1, kernel_size=3, padding=3//2)
    lam = 0.000001
    def forward(self, x):
        x = self.begin(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.head(x)
        return x.squeeze(1)

class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=7//2)
        self.conv2 = nn.Conv2d(in_ch, in_ch//3, kernel_size=5, padding=5//2)

        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=7//2)
        self.conv4 = nn.Conv2d(in_ch, in_ch//3, kernel_size=5, padding=5//2)

        self.skipconnections = nn.Conv2d(in_ch, in_ch//3, kernel_size=1)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, x):
        b, l, h, w = x.shape
        fx = self.conv2(self.selu(self.conv1(x)))

        lx = F.interpolate(x, (h//4, w//4))
        lx = self.conv4(self.selu(self.conv3(lx)))
        lx = F.interpolate(lx, (h, w))

        skip = self.skipconnections(x)
        return self.selu(torch.cat((fx, lx, skip), dim=1))

###############################################
#               Optimizer                     #
###############################################

class OptimizeParameters(nn.Module):
    def __init__(self, surface, lights, camera,
                 shadowing = True,
                 rough=0.5, diffuse=0.5, reflectance=0.5):
        '''
        initialization of class OptimizeParameters
        :param surface: (B, H, W), surface matrix in pixel-to-height representation
        :param lights: (tuple) -> (lights, boolean), lights: (L,3) light positions for all 12 light sources, boolean: if True lights is Parameter else is not a Parameter
        :param camera: (1, 1, 1, 1, 3), camera position
        :param shadowing: (boolean), if True shadow effects are applied to output of Filament renderer
        :param rough: (int), material parameter
        :param diffuse: (int), material parameter
        :param reflectance: (int), material parameter
        '''
        super().__init__()

        # scene parameters
        self.surface = Parameter(surface)
        self.lights_origin = lights[0]
        self.lights = Parameter(lights[0]) if lights[1] else lights[0]
        self.camera = camera

        # material parameters
        self.rough = Parameter(torch.tensor(rough))
        self.diffuse = Parameter(torch.tensor(diffuse))
        self.reflectance = Parameter(torch.tensor(reflectance))

        # gaussian filtered median
        self.gfm = getGfm()

        # shadow effects
        self.shadowing = shadowing
        self.shadow = None

        # relevant values for plotting
        self.errs = []
        self.errors = []
        self.roughs = []
        self.diffuses = []
        self.reflectances = []
        self.l_to_origin = []
        self.l_to_zero = []

    def forward(self):
        '''
        forward rendering function with applying Filament Renderer.
        :return: if shadowing=True the function outputs a rendered pytorch tensor multiplied with shadow effects, else the function outputs a rendered pytorch tensor without applying shadow effects.
        '''
        device = self.surface.device
        surface = self.surface - torch.mean(self.surface)
        output = filament_renderer(surface, self.camera.to(device), self.lights,
                                 rough=self.rough, diffuse=self.diffuse, reflectance=self.reflectance)
        if self.shadow is None:
            output0 = filament_renderer(surface, self.camera.to(device), self.lights,
                                       rough=0.2, diffuse=0.2, reflectance=0.2)
            self.shadow = (self.gfm.to(device) / output0).detach()
        if self.shadowing:
            return (output * self.shadow).squeeze(-1)
        else:
            return output.squeeze(-1)

    def plotImageComparism(self, samples, pred, path):
        '''
        save 12 images, which demonstrates the comparism of a real cabin-cap image with the prediction output of the Filament Renderer
        :param samples: (B, 1, H, W, 12), real cabin-cap image samples in pytorch tensor type
        :param pred: (B, 1, H, W, 12), prediction of cabin-cap samples (output of Filament renderer)
        :param path: (str), directory path for saving images
        :return: None
        '''
        for L in range(samples.shape[4]):
            p = cv2.cvtColor(pred[0, 0, ..., L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
            t = cv2.cvtColor(samples[0, 0, ..., L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(t)
            plt.title('real sample')
            plt.clim(0, 1.0)

            plt.subplot(1, 2, 2)
            plt.imshow(p)
            plt.title('rendered sample')
            plt.clim(0, 1.0)

            plt.savefig(os.path.join(path, f'TrueRGB-{L}.png'))
            plt.close()
    def plotDiagrams(self, plot_every, path):
        '''
        plot
        angles.png -> image, where every pixel value demonstrates the angle between normal vector and a vector in z-direction (0, 0, 1),
        error.png -> error while optimization,
        height-profile.png -> a height profile in x- and y-direction,
        l_to_origin.png -> line diagram with 12 lines corresponding to the 12 light sources, every line illustrates the distance between origin (position in construction plan) and actual optimized position,
        l_to_zero.png -> same as l_to_origin.png, but instead of origin distance is compared to position (0, 0, 0),
        material-parameters.png -> change of material parameters while optimization
        and save them to the given directory path.
        :param plot_every: (int), plotting period
        :param path: (str), string path, where plots are saved
        :return: None
        '''
        device = self.surface.device
        self.l_to_origin.append(
            torch.linalg.norm(self.lights_origin .cpu().detach() - self.lights.cpu().detach(), axis=-1).tolist())
        self.l_to_zero.append(torch.linalg.norm(self.lights.cpu().detach(), axis=-1).tolist())

        x = np.linspace(0, len(self.l_to_origin) - 1, len(self.l_to_origin)) * plot_every

        for L in range(12):
            plt.plot(x, np.array(self.l_to_origin)[:, L], label=f'{L}')

        plt.xlabel('iteration')
        plt.ylabel('distance to origin')
        plt.legend()
        plt.savefig(os.path.join(path, 'l_to_origin.png'))
        plt.close()

        x = np.linspace(0, len(self.l_to_zero) - 1, len(self.l_to_zero)) * plot_every

        for L in range(12):
            plt.plot(x, np.array(self.l_to_zero)[:, L], label=f'{L}')

        plt.xlabel('iteration')
        plt.ylabel('distance to zero')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(path, 'l_to_zero.png'))
        plt.close()

        x = np.linspace(0, len(self.errors) - 1, len(self.errors)) * plot_every
        plt.plot(x, self.errors, label='errors')
        plt.xlabel('iteration')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(os.path.join(path, f'error.png'))
        plt.close()

        height_profile_x_pred, height_profile_y_pred = getHeightProfile(self.surface, divide_by_mean=False)

        x = np.linspace(0, len(height_profile_x_pred) - 1, len(height_profile_x_pred))
        y = np.linspace(0, len(height_profile_y_pred) - 1, len(height_profile_y_pred))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)

        plt.plot(x, height_profile_x_pred, label='prediction')
        plt.xlabel('pixels')
        plt.ylabel('height')
        plt.legend()
        plt.title('profile in x-direction')

        plt.subplot(1, 2, 2)

        plt.plot(y, height_profile_y_pred, label='prediction')
        plt.xlabel('pixels')
        plt.ylabel('height')
        plt.legend()
        plt.title('profile in y-direction')

        plt.savefig(os.path.join(path, f'height-profile.png'))
        plt.close()

        normal_vectors = getNormals(self.surface.detach())
        z_vector = torch.tensor([0., 0., 1.]).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        angles = torch.acos((z_vector * normal_vectors).sum(dim=-1, keepdim=True)) * 90 / (torch.pi / 2)

        plt.imshow(angles[0, 0, ..., 0].cpu().detach().numpy())
        plt.colorbar()
        plt.savefig(os.path.join(path, f'angles.png'))
        plt.close()

        x = np.linspace(0, len(self.roughs) - 1, len(self.roughs)) * plot_every
        plt.plot(x, self.roughs, label='rough', color='red')
        plt.plot(x, self.diffuses, label='diffuse', color='green')
        plt.plot(x, self.reflectances, label='reflectance', color='blue')

        plt.xlabel('iterations')
        plt.ylabel('value')
        plt.legend()

        plt.savefig(os.path.join(path, f'material-parameters.png'))
        plt.close()

    def createParametersFile(self, path, selected_lights='all levels'):
        '''
        create parameters.txt file in directory path
        :param path: (str), string path, where textfile is saved
        :param selected_lights: (str), if
        'all levels': optimization happens for all light sources levels,
        'level 1': optimization happens only for level 1 light sources,
        'level 2': optimization happens only for level 2 light sources,
        'level 3': optimization happens only for level 3 light sources,
        'level 2+3': optimization happens only for level 2+3 light sources,
        oll other levels are neglected.
        :return: None
        '''
        parameters = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                parameters.append(name)

        with open(os.path.join(path, 'parameters.txt'), 'w') as f:
            f.write(f'Parameters {parameters}\n'
                    f'Rough {self.rough.item()} Diffuse {self.diffuse.item()} Reflectance {self.reflectance.item()} \n'
                    f'Camera {self.camera.detach()}\n'
                    f'Lights {self.lights.detach()}\n'
                    f'Surface Max {self.surface.detach().max()}'
                    f'Surface min {self.surface.detach().min()}\n'
                    f'AVG Err {statistics.mean(self.errs[-10:])}\n'
                    f'Difference lights {torch.linalg.norm(self.lights_origin.cpu() - self.lights.cpu().detach(), axis=-1)}\n'
                    f'Optimization with lights: {selected_lights}')
    
    def saveParameters(self, path):
        '''
        save all relevant parameters in given directory path
        :param path: (str), string path, where parameters are saved
        :return: None
        '''
        torch.save(self.rough.detach().cpu(), os.path.join(path, 'rough.pt'))
        torch.save(self.diffuse.detach().cpu(), os.path.join(path, 'diffuse.pt'))
        torch.save(self.reflectance.detach().cpu(), os.path.join(path, 'reflectance.pt'))
        torch.save(self.camera.detach().cpu(), os.path.join(path, 'camera.pt'))
        torch.save(self.lights.detach().cpu(), os.path.join(path, 'lights.pt'))
        torch.save(self.surface.detach(), os.path.join(path, 'surface.pt'))
        torch.save(self.shadow.detach(), os.path.join(path, 'shadow.pt'))


###############################################
#                  UNet                       #
###############################################

# https://amaarora.github.io/2020/09/13/unet.html

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(12, 24, 48)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    def __init__(self, enc_chs=(12, 24, 48, 96, 192), dec_chs=(192, 96, 48, 24), num_class=1, retain_dim=True, out_sz=(512, 512)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out.squeeze(1)