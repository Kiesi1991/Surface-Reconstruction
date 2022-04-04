from torch import nn
import torch
import torchvision
import torch.nn.functional as F

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
        self.begin = nn.Conv2d(8, 12, kernel_size=1)
        self.res_blocks = nn.ModuleList([ResBlock(12) for i in range(layers)])
        self.head = nn.Conv2d(12, 1, kernel_size=3, padding=3//2)

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
from filament_renderer import filament_renderer

class OptimizeParameters(nn.Module):
    def __init__(self, mesh, lights, camera):
        super().__init__()
        self.mesh = nn.parameter.Parameter(mesh)
        self.lights = nn.parameter.Parameter(lights)
        self.camera = nn.parameter.Parameter(camera)

        self.rough = nn.parameter.Parameter(torch.normal(mean=torch.tensor(0.35), std=torch.tensor(0.1)))
        self.diffuse = nn.parameter.Parameter(torch.normal(mean=torch.tensor(0.55), std=torch.tensor(0.1)))
        self.f0P = nn.parameter.Parameter(torch.normal(mean=torch.tensor(0.67), std=torch.tensor(0.1)))
        self.light_intensity = nn.parameter.Parameter(torch.ones((1, 1, 1, 1, 12, 1)))
        self.light_color = nn.parameter.Parameter(torch.ones((1, 1, 1, 1, 12, 1)))
        #self.inv_falloff = nn.parameter.Parameter(torch.ones(1, 1, 1, 1, 7, 1) * 0.005)
        self.x = nn.parameter.Parameter(torch.tensor(1.202888))
        self.y = nn.parameter.Parameter(torch.tensor(1.608325))
    def forward(self):
        color = filament_renderer(self.mesh, self.camera, self.lights,
                                 rough=self.rough, diffuse=self.diffuse, light_intensity=self.light_intensity, light_color=self.light_color, f0P=self.f0P, x=self.x, y=self.y)
        return color.squeeze(0).squeeze(0).squeeze(-1)



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