import torch

from utils import *
from filament_renderer import filament_renderer

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
                 camera, lights, light_intensity, intensity,
                 rough,
                 diffuse,
                 reflectance ,
                 x=1.6083,
                 y=1.20288,
                 device='cpu'):
        self.rough = torch.clamp(rough, min=0., max=1.).to(device)
        self.diffuse = torch.clamp(diffuse, min=0., max=1.).to(device)
        self.reflectance = torch.clamp(reflectance, min=0., max=1.).to(device)
        #self. camera = camera.to(device)
        self.camera = camera.to(device)
        #self.lights = lights.to(device)
        self.lights = lights.to(device)
        self.device = device
        self.x = x
        self.y = y
        self.light_intensity = light_intensity.to(device) * intensity.to(device)
        self.light_color = torch.ones_like(self.light_intensity).to(device)

        self.la = get_light_attenuation().to(device)
    def forward(self, surface):
        color = filament_renderer(surface, camera=self.camera, lights=self.lights,la=self.la, light_intensity=self.light_intensity, light_color=self.light_color, rough=self.rough, diffuse=self.diffuse, reflectance=self.reflectance, x=self.x, y=self.y).permute(0, 4, 2, 3, 1, 5)[None].squeeze(0).squeeze(-1).squeeze(-1)
        return color

