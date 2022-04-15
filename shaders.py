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
                    rough=0.7913722395896912,
                    diffuse=1.4271979331970215,
                    f0P=1.3222495317459106 ,
                    x=1.2028881311416626,
                    y=1.6083250045776367,
                    device='cpu'):
        self.rough = rough
        self.diffuse = diffuse
        self. f0P = f0P
        self. camera = camera.to(device)
        self.lights = lights.to(device)
        self.device = device
        self.x = x
        self.y = y
        self.light_intensity = torch.tensor([[[[[[0.6317],
            [0.5862],
            [0.5012],
            [0.6062],
            [1.4130],
            [1.2370],
            [0.9007],
            [0.6430],
            [2.9547],
            [2.6376],
            [2.1030],
            [2.8895]]]]]])
        self.light_color = torch.ones_like(self.light_intensity)

        self.la = get_light_attenuation().to(device)

    def forward(self, surface):
        color = filament_renderer(surface, camera=self.camera, lights=self.lights,la=self.la, light_intensity=self.light_intensity, light_color=self.light_color, rough=self.rough, diffuse=self.diffuse, f0P=self.f0P, x=self.x, y=self.y).permute(0, 4, 2, 3, 1, 5)[None].squeeze(0).squeeze(-1).squeeze(-1)
        return color

