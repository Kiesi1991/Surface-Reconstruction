from torch import nn
from torch.nn.parameter import Parameter
import torch
import torchvision
import torch.nn.functional as F
from filament_renderer import filament_renderer
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
        #ftrs = []
        x = self.begin(x)
        for block in self.res_blocks:
            x = block(x)
            #ftrs.append(x)
        x = self.head(x)
        return x.squeeze(1)
        #return torch.clamp(x.squeeze(1), min=-0.5, max=0.5)

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
    def __init__(self, surface, lights, camera, gfm,
                 shadowing=True, par_li=False,
                 par_r=True, par_d=True, par_ref=True,
                 device='cpu',
                 intensity=1.,
                 rough=0.5, diffuse=0.5, reflectance=0.5):
        super().__init__()


        self.mesh = Parameter(surface[0].to(device)) if surface[1] else surface[0].to(device)
        self.lights_origin = lights[0]
        self.lights = Parameter(lights[0].to(device)) if lights[1]  else lights[0].to(device)
        self.camera = Parameter(camera[0].to(device)) if camera[1] else camera[0].to(device)


        light_intensity = torch.ones((1,1,1,1,12,1)).to(device)
        self.light_intensity = Parameter(light_intensity) if par_li else light_intensity
        self.light_color = torch.ones_like(self.light_intensity).to(device)
        self.intensity = torch.tensor(intensity).to(device)

        self.rough = Parameter(torch.tensor(rough).to(device)) if par_r else torch.tensor(rough).to(device)
        self.diffuse = Parameter(torch.tensor(diffuse).to(device)) if par_d else torch.tensor(diffuse).to(device)
        self.reflectance = Parameter(torch.tensor(reflectance).to(device)) if par_ref else torch.tensor(reflectance).to(device)

        self.gfm = gfm.to(device)

        self.device = device

        self.shadowing = shadowing
        self.shadow = None

        # values for plotting
        self.errs = []
        self.errors = []
        self.roughs = []
        self.diffuses = []
        self.reflectances = []
        self.intensities = []
        self.l_to_origin = []
        self.l_to_zero = []

    def forward(self):
        rough = torch.clamp(self.rough, min=0., max=1.)
        diffuse = torch.clamp(self.diffuse, min=0., max=1.)

        light_intensity = self.light_intensity * self.intensity

        mesh = self.mesh - torch.mean(self.mesh)

        color = filament_renderer(mesh, self.camera, self.lights,
                                 rough=rough, diffuse=diffuse, light_intensity=light_intensity, light_color=self.light_color, reflectance=self.reflectance)

        if self.shadowing and self.shadow is None:
            self.shadow = (self.gfm / color).detach()

        if self.shadowing:
            return (color * self.shadow).squeeze(-1)
        else:
            return color.squeeze(-1)

    def plotImageComparism(self, samples, pred, path):
        for L in range(samples.shape[4]):
            p = cv2.cvtColor(pred[0, 0, ..., L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
            t = cv2.cvtColor(samples[0, 0, ..., L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(t)
            plt.clim(0, 1.0)

            plt.subplot(1, 2, 2)
            plt.imshow(p)
            plt.clim(0, 1.0)

            plt.savefig(os.path.join(path, f'TrueRGB-{L}.png'))
            plt.close()
    def plotDiagrams(self, model, plot_every, path, synthetic,
                     rough_origin=0, reflectance_origin=0, diffuse_origin=0):
        model.l_to_origin.append(
            torch.linalg.norm(self.lights_origin .cpu().detach() - model.lights.cpu().detach(), axis=-1).tolist())
        model.l_to_zero.append(torch.linalg.norm(model.lights.cpu().detach(), axis=-1).tolist())

        x = np.linspace(0, len(model.l_to_origin) - 1, len(model.l_to_origin)) * plot_every

        for L in range(12):
            plt.plot(x, np.array(model.l_to_origin)[:, L], label=f'{L}')

        plt.xlabel('iteration')
        plt.ylabel('distance to origin')
        plt.legend()
        plt.savefig(os.path.join(path, 'l_to_origin.png'))
        plt.close()

        x = np.linspace(0, len(model.l_to_zero) - 1, len(model.l_to_zero)) * plot_every

        for L in range(12):
            plt.plot(x, np.array(model.l_to_zero)[:, L], label=f'{L}')

        plt.xlabel('iteration')
        plt.ylabel('distance to zero')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(path, 'l_to_zero.png'))
        plt.close()

        x = np.linspace(0, len(model.errors) - 1, len(model.errors)) * plot_every
        plt.plot(x, model.errors, label='errors')
        plt.xlabel('iteration')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(os.path.join(path, f'error.png'))
        plt.close()

        if synthetic:
            height_profile_x_gt, height_profile_y_gt = getHeightProfile(self.mesh)
        height_profile_x_pred, height_profile_y_pred = getHeightProfile(model.mesh)

        x = np.linspace(0, len(height_profile_x_pred) - 1, len(height_profile_x_pred))
        y = np.linspace(0, len(height_profile_y_pred) - 1, len(height_profile_y_pred))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)

        if synthetic:
            plt.plot(x, height_profile_x_gt, label='ground truth')
        plt.plot(x, height_profile_x_pred, label='prediction')
        plt.xlabel('pixels')
        plt.ylabel('height')
        plt.legend()
        plt.title('profile in x-direction')

        plt.subplot(1, 2, 2)

        if synthetic:
            plt.plot(y, height_profile_y_gt, label='ground truth')
        plt.plot(y, height_profile_y_pred, label='prediction')
        plt.xlabel('pixels')
        plt.ylabel('height')
        plt.legend()
        plt.title('profile in y-direction')

        plt.savefig(os.path.join(path, f'height-profile.png'))
        plt.close()

        normal_vectors = getNormals(model.mesh.detach())
        z_vector = torch.tensor([0., 0., 1.]).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        angles = torch.acos((z_vector * normal_vectors).sum(dim=-1, keepdim=True)) * 90 / (torch.pi / 2)

        plt.imshow(angles[0, 0, ..., 0].cpu().detach().numpy())
        plt.colorbar()
        plt.savefig(os.path.join(path, f'angles.png'))
        plt.close()

        if synthetic:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)

        x = np.linspace(0, len(model.roughs) - 1, len(model.roughs)) * plot_every
        plt.plot(x, model.roughs, label='rough', color='red')
        plt.plot(x, model.diffuses, label='diffuse', color='green')
        plt.plot(x, model.reflectances, label='reflectance', color='blue')

        plt.xlabel('iterations')
        plt.ylabel('value')
        if synthetic:
            plt.plot(x, [rough_origin] * len(model.roughs), color='red', linestyle='dashed')
            plt.plot(x, [diffuse_origin] * len(model.diffuses), color='green', linestyle='dashed')
            plt.plot(x, [reflectance_origin] * len(model.reflectances), color='blue', linestyle='dashed')
            plt.title(f'parameters with constant intensity {model.intensity.cpu().detach().numpy().item()}\n'
                      f'Synthetic: (rough,diffuse,reflectance)={(rough_origin, diffuse_origin, reflectance_origin)}; \n'
                      f'Initial value vor prediction: (rough,diffuse,reflectance)={(rough_origin, diffuse_origin, reflectance_origin)}')
        else:
            plt.title(f'parameters with constant intensity {model.intensity.cpu().detach().numpy().item()}')
        plt.legend()

        if synthetic:
            plt.subplot(1, 2, 2)

            surface_line = self.mesh.cpu().detach().numpy()[0, 200, :]
            pred_surface_line = model.mesh.cpu().detach().numpy()[0, 200, :]

            x = np.linspace(0, len(surface_line) - 1, len(surface_line))
            plt.plot(x, surface_line, label='ground truth')
            plt.plot(x, pred_surface_line, label='prediction')
            plt.xlabel('x')
            plt.ylabel('height')
            plt.legend()

        plt.savefig(os.path.join(path, f'material-parameters.png'))
        plt.close()

    def createParametersFile(self, path, selected_lights='all levels'):
        parameters = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                parameters.append(name)

        with open(os.path.join(path, 'parameters.txt'), 'w') as f:
            f.write(f'Parameters {parameters}\n'
                    f'Rough {self.rough.item()} Diffuse {self.diffuse.item()} Reflectance {self.reflectance.item()} \n'
                    f'Camera {self.camera.detach()}\n'
                    f'Lights {self.lights.detach()}\n'
                    f'Surface Max {self.mesh.detach().max()}'
                    f'Surface min {self.mesh.detach().min()}\n'
                    f'Light Intensity {self.light_intensity.detach()}\n'
                    f'Intensity {self.intensity.detach()}\n'
                    f'AVG Err {statistics.mean(self.errs[-10:])}\n'
                    f'Difference lights {torch.linalg.norm(self.lights_origin.cpu() - self.lights.cpu().detach(), axis=-1)}\n'
                    f'Optimization with lights: {selected_lights}')


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