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
                    camera, lights,
                    rough=0.34326401352882385,
                    diffuse=0.7738519906997681,
                    f0P=0.48454031348228455,
                    x=1.2918046712875366,
                    y=1.454910159111023,
                    device='cpu'):
        self.rough = rough
        self.diffuse = diffuse
        self. f0P = f0P
        self. camera = camera.to(device)
        self.lights = lights.to(device)
        self.device = device
        self.x = x
        self.y = y
        self.light_intensity = torch.tensor([[[[[[1.4256],
            [1.3992],
            [1.3907],
            [1.4349],
            [1.2344],
            [1.2184],
            [1.1475],
            [0.9217],
            [0.8218],
            [0.7301],
            [0.6995],
            [0.7970]]]]]])
        self.light_color = torch.tensor([[[[[[1.4256],
            [1.3992],
            [1.3907],
            [1.4349],
            [1.2344],
            [1.2184],
            [1.1475],
            [0.9217],
            [0.8218],
            [0.7301],
            [0.6995],
            [0.7970]]]]]])

    def forward(self, surface):
        color = filament_renderer(surface, camera=self.camera, lights=self.lights, light_intensity=self.light_intensity, light_color=self.light_color, rough=self.rough, diffuse=self.diffuse, f0P=self.f0P, x=self.x, y=self.y).permute(0, 4, 2, 3, 1, 5)[None].squeeze(0).squeeze(-1).squeeze(-1)
        return color

