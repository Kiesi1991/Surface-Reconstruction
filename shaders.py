import torch
from utils import *

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
                 reflectance,
                 shadow,
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

        self.shadow = shadow.squeeze(0).squeeze(-1).permute(0,3,1,2)
    def forward(self, surface):
        color = filament_renderer(surface, camera=self.camera, lights=self.lights, light_intensity=self.light_intensity, light_color=self.light_color, rough=self.rough, diffuse=self.diffuse, reflectance=self.reflectance, x=self.x, y=self.y).permute(0, 4, 2, 3, 1, 5)[None].squeeze(0).squeeze(-1).squeeze(-1)
        return color * self.shadow


class SyntheticSamples():
    def __init__(self, samples, lights, camera,
                 shadow,
                 rough=0.11, diffuse=0.19, reflectance=0.15):
        # scene parameters
        if torch.cuda.is_available():
            self.surface = createSurface(resolution=(samples.shape[2], samples.shape[3])).to('cuda').unsqueeze(0)
        else:
            self.surface = createSurface(resolution=(samples.shape[2], samples.shape[3])).unsqueeze(0)
        self.lights = lights + torch.normal(mean=0, std=0.2, size=lights.shape)
        self.camera = camera

        # material parameters
        self.rough = rough + np.random.normal(0, 0.02)
        self.diffuse = diffuse + np.random.normal(0, 0.02)
        self.reflectance = reflectance + np.random.normal(0, 0.02)

        # shadow effects
        self.shadow = shadow
    def forward(self):

        device = self.surface.device
        surface = self.surface - torch.mean(self.surface)
        output = filament_renderer(surface, self.camera.to(device), self.lights.to(device),
                                 rough=self.rough, diffuse=self.diffuse, reflectance=self.reflectance)
        return output * self.shadow

