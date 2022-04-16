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
                    rough=0.6826966404914856,
                    diffuse=1.0560709238052368,
                    f0P=0.9894510507583618 ,
                    x=1.2028881311416626,
                    y=1.6083250045776367,
                    device='cpu'):
        self.rough = rough
        self.diffuse = diffuse
        self. f0P = f0P
        #self. camera = camera.to(device)
        self.camera = torch.tensor([[[[[-0.4919,  0.8713,  7.7930]]]]]).to(device)
        #self.lights = lights.to(device)
        self.lights = torch.tensor([[ 0.4610, -2.3101,  5.4689],
        [ 2.0637, -0.1640,  5.8244],
        [ 0.0524,  2.2153,  5.6867],
        [-1.9160,  0.4533,  5.9731],
        [ 0.2925, -2.0323,  3.9218],
        [ 2.0486, -0.1973,  3.8636],
        [ 0.0061,  2.4624,  3.6414],
        [-2.2224,  0.4564,  3.7725],
        [ 0.1593, -2.2010,  1.5920],
        [ 2.2989,  0.0636,  1.4572],
        [-0.0257,  2.6306,  1.1837],
        [-2.0910,  0.2743,  1.6005]]).to(device)
        self.device = device
        self.x = x
        self.y = y
        self.light_intensity = torch.tensor([[[[[[0.8111],
            [0.7329],
            [0.6446],
            [0.7719],
            [1.5840],
            [1.3656],
            [1.0796],
            [0.7442],
            [1.9819],
            [1.7916],
            [1.6281],
            [1.9292]]]]]])
        self.light_color = torch.ones_like(self.light_intensity)

        self.la = get_light_attenuation().to(device)

    def forward(self, surface):
        color = filament_renderer(surface, camera=self.camera, lights=self.lights,la=self.la, light_intensity=self.light_intensity, light_color=self.light_color, rough=self.rough, diffuse=self.diffuse, f0P=self.f0P, x=self.x, y=self.y).permute(0, 4, 2, 3, 1, 5)[None].squeeze(0).squeeze(-1).squeeze(-1)
        return color

