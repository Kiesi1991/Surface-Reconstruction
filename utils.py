from scipy.ndimage import gaussian_filter
import numpy as np
import torch

def createSurface(resolution):
    surface = np.random.choice(np.array([1.0, -1.0, 0.0]), size=resolution, p=[0.01, 0.01, 0.98])
    surface = gaussian_filter(surface, sigma=10, mode='reflect')
    return torch.from_numpy(surface)

def getNormals(surface, x=4, y=2):
    hx = x / surface.shape[1]
    hy = y / surface.shape[0]

    dfdx = (surface[..., 1:] - surface[..., :-1]) / hx
    dfdx1 = ((surface[:, -1] - surface[:, -2]) / hx).unsqueeze(1)
    dfdx = torch.cat((dfdx, dfdx1), dim=1).unsqueeze(2)

    dfdy = ((surface[1:, ...] - surface[:-1, ...]) / hy)
    dfdy1 = ((surface[-1, :] - surface[-2, :]) / hx).unsqueeze(0)
    dfdy = torch.cat((dfdy, dfdy1), dim=0).unsqueeze(2)

    z = torch.ones_like(dfdx)

    normals = torch.cat((-dfdx, -dfdy, z), dim=2)

    return normalize(normals).unsqueeze(0)

def getVectors(surface, targetLocation, x, y):

    h, w = surface.shape
    dx = x/w
    dy = y/h

    X = np.expand_dims(np.linspace(-(w // 2) * dx, (w // 2) * dx, num=w), axis=0)
    Y = np.expand_dims(np.linspace(-(h // 2) * dy, (h // 2) * dy, num=h), axis=1)

    X = torch.from_numpy(np.expand_dims(np.repeat(X, h, axis=0), axis=2))
    Y = torch.from_numpy(np.expand_dims(np.repeat(Y, w, axis=1), axis=2))
    Z = surface.unsqueeze(2)

    surfacePoints = torch.cat((X, Y, Z), dim=2).unsqueeze(0)


    V = targetLocation-surfacePoints

    return normalize(V)

def normalize(vector):
    Norms = torch.linalg.norm(vector, axis=2, keepdims=True)
    return vector/Norms