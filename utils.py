from scipy.ndimage import gaussian_filter
import numpy as np
import torch

#[(4, 0.0001),
#                                    (6, 0.0005),
#                                    (8, 0.0005),
#                                    (10, 0.001),
#                                    (20, 0.001),
#                                    (50, 0.001),
#                                    (100, 0.001)]

def createSurface(resolution, para=[(30, 0.01)]):
    surface = np.zeros(resolution)
    for sigma, p in para:
        surface1 = np.random.choice(np.array([1.0, 0.0]), size=resolution, p=[p, 1.0-p])
        var = np.random.normal(0, sigma//10, 1)[0]
        var2 = np.random.normal(0, 0.005, 1)[0]
        surface1 = gaussian_filter(surface1, sigma=sigma+var, mode='reflect')
        surface += (surface1 / surface1.max()) * (0.003+var2) * ((sigma+var)/5)
    return torch.from_numpy(surface)

def getNormals(surface, x=4, y=2):

    hx = x / surface.shape[2]
    hy = y / surface.shape[1]

    dfdx = (surface[..., 1:] - surface[..., :-1]) / hx
    dfdx1 = ((surface[..., -1] - surface[..., -2]) / hx).unsqueeze(2)
    dfdx = torch.cat((dfdx, dfdx1), dim=2).unsqueeze(3)

    dfdy = ((surface[:, 1:, ...] - surface[:, :-1, ...]) / hy)
    dfdy1 = ((surface[:, -1, ...] - surface[:, -2, ...]) / hx).unsqueeze(1)
    dfdy = torch.cat((dfdy, dfdy1), dim=1).unsqueeze(3)

    z = torch.ones_like(dfdx)

    normals = torch.cat((-dfdx, -dfdy, z), dim=3)

    return normalize(normals.unsqueeze(1))

def getVectors(surface, targetLocation, x, y, norm=True):

    device = surface.device

    b, h, w = surface.shape
    dx = x/w
    dy = y/h

    X = torch.linspace(-(w // 2), (w // 2), steps=w).unsqueeze(0) * dx
    Y = torch.linspace(-(h // 2), (h // 2), steps=h).unsqueeze(1) * dy

    X = X.repeat(h, 1).unsqueeze(0).repeat(b, 1, 1).unsqueeze(-1).to(device)
    Y = Y.repeat(1, w).unsqueeze(0).repeat(b, 1, 1).unsqueeze(-1).to(device)
    Z = surface.unsqueeze(3)

    surfacePoints = torch.cat((X, Y, Z), dim=3).unsqueeze(1)

    V = targetLocation - surfacePoints
    if norm:
        return normalize(V)
    else:
        return V

def normalize(vector):
    Norms = torch.linalg.norm(vector, axis=4, keepdims=True)
    return vector/Norms