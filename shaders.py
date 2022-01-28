from utils import *

class PhongShading():
    def __init__(self,
                    camera, lights,
                    length, width,
                    specular=0.8, diffuse=0.8,
                    ambient=0.5, shininess=50):
        self.specular = specular
        self.diffuse = diffuse
        self. ambient = ambient
        self. shininess = shininess
        self. camera = torch.tensor([[[camera]]])
        self.length, self.width = length, width
        self.lights = torch.tensor(lights).unsqueeze(1).unsqueeze(1)
    def forward(self, surface):
        # calculating normalized vectors for phong shading
        N = getNormals(surface, x=self.length, y=self.width)
        V = getVectors(surface, self.camera, x=self.length, y=self.width)
        L = getVectors(surface, self.lights, x=self.length, y=self.width)
        LoN = torch.einsum('abcd, abcd -> abc', L, N).squeeze(0)
        R = 2 * LoN.unsqueeze(2) * N.squeeze(0) - L.squeeze(0)
        RoV = torch.einsum('abcd, abcd -> abc', R.unsqueeze(0), V).squeeze(0)

        I = self.ambient + self.diffuse * LoN + self.specular * (RoV ** self.shininess)

        return I / I.max()


