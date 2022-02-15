from scipy.ndimage import gaussian_filter
import numpy as np
import torch

def createSurface(resolution):
    surface = np.random.choice(np.array([1.0, -1.0, 0.0]), size=resolution, p=[0.01, 0.01, 0.98])
    surface = gaussian_filter(surface, sigma=10, mode='reflect')
    # varieren von sigma, for loop for random points, other filters, use more pixel
    return torch.from_numpy(surface)

def getNormals(surface, x=4, y=2):

    hx = x / surface.shape[2]
    hy = y / surface.shape[1]

    # alternative: torch.gradient or torch.diff
    dfdx = (surface[..., 1:] - surface[..., :-1]) / hx
    dfdx1 = ((surface[..., -1] - surface[..., -2]) / hx).unsqueeze(2)
    dfdx = torch.cat((dfdx, dfdx1), dim=2).unsqueeze(3)

    dfdy = ((surface[:, 1:, ...] - surface[:, :-1, ...]) / hy)
    dfdy1 = ((surface[:, -1, ...] - surface[:, -2, ...]) / hx).unsqueeze(1)
    dfdy = torch.cat((dfdy, dfdy1), dim=1).unsqueeze(3)

    z = torch.ones_like(dfdx)

    normals = torch.cat((-dfdx, -dfdy, z), dim=3)

    return normalize(normals.unsqueeze(1))

def getVectors(surface, targetLocation, x, y):

    device = surface.device

    b, h, w = surface.shape
    dx = x/w
    dy = y/h

    X = torch.linspace(-(w // 2) * dx, (w // 2) * dx, steps=w).unsqueeze(0)
    Y = torch.linspace(-(h // 2) * dy, (h // 2) * dy, steps=h).unsqueeze(1)

    X = X.repeat(h, 1).unsqueeze(0).repeat(b, 1, 1).unsqueeze(-1).to(device)
    Y = Y.repeat(1, w).unsqueeze(0).repeat(b, 1, 1).unsqueeze(-1).to(device)
    Z = surface.unsqueeze(3)

    surfacePoints = torch.cat((X, Y, Z), dim=3).unsqueeze(1)

    V = targetLocation - surfacePoints

    return normalize(V)

def normalize(vector):
    Norms = torch.linalg.norm(vector, axis=4, keepdims=True)
    return vector/Norms