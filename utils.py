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

def createSurface(resolution):
    surface = np.zeros(resolution)

    p = 0.4
    h = 0.01

    sigmas = [2.5,5,10,15]

    for sigma in sigmas:
        x_var = np.random.normal(0, sigma*0.2, 1)[0]
        y_var = np.random.normal(0, sigma*0.2, 1)[0]
        p_var = np.clip(np.random.normal(0, p * 0.8, 1)[0], 0.00001, 0.05)
        surface1 = np.zeros_like(surface)
        for _ in range(3):
            surface1 += random_walk(size=resolution, p=p, p_var=p_var)
        surface1 = np.clip(surface1, 0.0, 1.0)
        #surface1 = np.random.choice(np.array([1.0, 0.0]), size=resolution, p=[(p+p_var), 1.0 - (p+p_var)])
        surface1 = gaussian_filter(surface1, sigma=sigma+x_var+y_var, mode='reflect')
        surface1 /= surface1.max()
        surface1 *= (sigma+(x_var+y_var))
        surface += surface1

    surface -= surface.min()
    surface /= surface.max()
    h_var = np.random.normal(0, h*0.2, 1)[0]
    surface *= (h+h_var)
    o_var = np.random.normal(0, h*0.1, 1)[0]
    surface -= ((surface.max() / 2.0)+o_var)

    return torch.from_numpy(surface)

    '''for sigma, p in para:
        surface1 = np.random.choice(np.array([1.0, 0.0]), size=resolution, p=[p, 1.0-p])
        #var = np.random.normal(0, sigma//10, 1)[0]
        surface1 = gaussian_filter(surface1, sigma=sigma, mode='reflect')
        surface += (surface1 / surface1.max())# * ((sigma)/5)
    #var2 = np.random.normal(0, 0.005, 1)[0]
    surface = surface - surface.min()
    surface = (surface / surface.max()) * (0.1)
    return torch.from_numpy(surface)'''

def random_walk(size, p, p_var):
    length = np.random.randint(100, size=1)[0] + 1
    mask = np.random.choice(np.array([1.0, 0.0]), size=size, p=[(p+p_var), 1.0 - (p+p_var)])
    actions = np.random.randint(4, size=length)+1 # actions: 1=right, 2=left, 3=up, 4=down
    h,w = size
    surface = np.zeros((h+length*2,w+length*2))
    surface[length:-length,length:-length] = mask
    x = 0
    y = 0
    for action in actions:
        if action == 1:
            x += 1
        if action == 2:
            x -= 1
        if action == 3:
            y -= 1
        if action == 4:
            y += 1
        surface[length+y:(h+length+y),length+x:(w+length+x)] += mask #kumlative summer, scatter, conv

    return np.clip(surface[length:-length,length:-length], 0.0, 1.0)



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

def surface_height_distance(surface):
    #heightDistance1 = surface[:,1:,:] - surface[:,:-1,:]
    #heightDistance2 = surface[:,:,1:] - surface[:,:,:-1]

    mse = torch.nn.MSELoss()
    err1 = mse(surface[:,1:,:], surface[:,:-1,:])
    err2 = mse(surface[:,:,1:], surface[:,:,:-1])
    return err1 + err2