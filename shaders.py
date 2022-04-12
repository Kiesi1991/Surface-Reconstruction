import torch

from utils import *
from filament_renderer import filament_renderer
from scanData import get_light_attenuation

class PhongShading():
    def __init__(self,
                    camera, lights,
                    length, width,
                    specular=0.8, diffuse=0.8,
                    ambient=0.5, shininess=50,
                    device='cpu'):
        self.specular = specular
        self.diffuse = diffuse
        self. ambient = ambient
        self. shininess = shininess
        self. camera = torch.tensor([[[camera]]]).to(device).unsqueeze(1) #1,1,1,1,3
        self.length, self.width = length, width
        self.lights = torch.tensor(lights).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device) #1,12,1,1,3
    def forward(self, surface):
        # calculating normalized vectors for phong shading
        N = getNormals(surface, x=self.length, y=self.width)
        V = getVectors(surface, self.camera, x=self.length, y=self.width)
        L = getVectors(surface, self.lights, x=self.length, y=self.width)
        LoN = torch.einsum('abcde, abcde -> abcd', L, N)
        R = 2 * LoN.unsqueeze(4) * N - L
        RoV = torch.einsum('abcde, abcde -> abcd', R, V)

        I = self.ambient + self.diffuse * LoN + self.specular * (RoV ** self.shininess)

        return (I / I.max()).float()

class FilamentShading():
    def __init__(self,
                    camera, lights,
                    rough=0.262628436088562,
                    diffuse=0.43219268321990967,
                    f0P=0.8909110426902771 ,
                    x=0.6803215146064758,
                    y=0.9094781875610352,
                    device='cpu'):
        self.rough = rough
        self.diffuse = diffuse
        self. f0P = f0P
        self. camera = camera.to(device)
        self.lights = lights.to(device)
        self.device = device
        self.x = x
        self.y = y
        self.light_intensity = torch.tensor([[[[[[1.7605],
            [1.6859],
            [1.5881],
            [1.7304],
            [1.4134],
            [1.4337],
            [1.3288],
            [1.1774],
            [1.1116],
            [1.0085],
            [0.8087],
            [1.0708]]]]]])
        self.light_color = torch.tensor([[[[[[1.7605],
            [1.6859],
            [1.5881],
            [1.7304],
            [1.4134],
            [1.4337],
            [1.3288],
            [1.1774],
            [1.1116],
            [1.0085],
            [0.8087],
            [1.0708]]]]]])

        self.la = get_light_attenuation().to(device)

    def forward(self, surface):
        color = filament_renderer(surface, camera=self.camera, lights=self.lights,la=self.la, light_intensity=self.light_intensity, light_color=self.light_color, rough=self.rough, diffuse=self.diffuse, f0P=self.f0P, x=self.x, y=self.y).permute(0, 4, 2, 3, 1, 5)[None].squeeze(0).squeeze(-1).squeeze(-1)
        return color

