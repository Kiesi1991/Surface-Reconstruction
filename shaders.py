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
    def __init__(self,optimized_parameters):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimized_surface = optimized_parameters['surface'].to(device)
        self.camera = optimized_parameters['camera'].to(device)
        self.lights = optimized_parameters['lights'].to(device)
        # optimized material parameters
        self.rough = optimized_parameters['rough'].to(device)
        self.diffuse = optimized_parameters['diffuse'].to(device)
        self.reflectance = optimized_parameters['reflectance'].to(device)
        # shadow effects
        self.shadow = optimized_parameters['shadow'].to(device)
    def forward(self, surface):
        color = filament_renderer(surface, camera=self.camera, lights=self.lights,
                                  rough=self.rough, diffuse=self.diffuse, reflectance=self.reflectance)
        return (color * self.shadow).permute(0,4,2,3,1,5).squeeze(-1).squeeze(-1)


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

