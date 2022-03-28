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
                    specular=0.8, diffuse=0.8,
                    ambient=0.5, shininess=50,
                    device='cpu'):
        self.specular = specular
        self.diffuse = diffuse
        self. ambient = ambient
        self. shininess = shininess
        self. camera = camera.to(device)
        self.lights = lights.to(device)
        self.device = device
    def forward(self, surface):
        color = filament_renderer(surface, self.camera, self.lights).permute(0, 4, 2, 3, 1, 5)[None].squeeze(0).squeeze(-1).squeeze(-1)
        return color

