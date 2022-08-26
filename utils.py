from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import os
import glob
import imageio
from PIL import Image
from torchvision import transforms
from datetime import datetime

def createSurface(resolution):
    surface = np.zeros(resolution)

    p = 0.02
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

    return torch.from_numpy(surface) - torch.mean(torch.from_numpy(surface))

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



def getNormals(surface, pd=0.0031):
    '''
    calculates normal vectors given a surface matrix.
    :param surface: (B, H, W), surface matrix in pixel-to-height representation, every entry contains a height value in z-direction
    :param pd: (float), distance between pixels
    :return: (B, 1, H, W, 3), normalized normal vectors for every pixel
    '''
    dfdx = (surface[..., 1:] - surface[..., :-1]) / pd
    dfdx1 = ((surface[..., -1] - surface[..., -2]) / pd).unsqueeze(2)
    dfdx = torch.cat((dfdx, dfdx1), dim=2).unsqueeze(3)

    dfdy = ((surface[:, 1:, ...] - surface[:, :-1, ...]) / pd)
    dfdy1 = ((surface[:, -1, ...] - surface[:, -2, ...]) / pd).unsqueeze(1)
    dfdy = torch.cat((dfdy, dfdy1), dim=1).unsqueeze(3)

    z = torch.ones_like(dfdx)
    normals = torch.cat((-dfdx, -dfdy, z), dim=3)
    return normalize(normals.unsqueeze(1))

def getVectors(surface, targetLocation, pd=0.0031, norm=True):
    '''
    calculates vectors between target positions and pixel positions.
    :param surface: (B, H, W), surface matrix in pixel-to-height representation, every entry contains a height value in z-direction
    :param targetLocation: (1, 1 or L, 1, 1, 3), target position for calculating vectors e.g. light positions, camera position
    :param pd: (float), distance between pixels
    :param norm: (boolean), if TRUE output is normalized
    :return: (normalized) vector between pixel positions and target position(s)
    '''
    device = surface.device
    b, h, w = surface.shape

    X = torch.linspace(-(w // 2), (w // 2), steps=w).unsqueeze(0).to(device) * pd
    Y = torch.linspace(-(h // 2), (h // 2), steps=h).unsqueeze(1).to(device) * pd

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
    '''
    normalize a vector at dimension -1
    :param vector: (..., 3) pytorch tensor, which should be normalized
    :return: (..., 3) normalized vector
    '''
    Norms = torch.linalg.norm(vector, axis=-1, keepdims=True)
    return vector/Norms

def scan(path_real_samples):
    convert_tensor = transforms.ToTensor()

    file_path = os.path.join(path_real_samples, '**', f'*.jpg')
    paths = glob.glob(file_path, recursive=True)

    images = {}
    for idx, path in enumerate(paths):
        number = path[-6:-4]
        if number[0] == '/':
            number = number[1]
        number = int(number)
        im = imageio.imread(path)
        im = convert_tensor(Image.fromarray(im))[0].unsqueeze(-1)
        if number not in images:
            images[number] = im
        else:
            images[number] = torch.cat((images[number], im), dim=-1)
    return images

def getGfm(path_real_samples='Sensor_2'):

    images = scan(path_real_samples)

    gfm = None
    for im_nr in sorted(images.keys()) :
        im = torch.from_numpy(gaussian_filter(torch.median(images[im_nr], dim=2)[0], sigma=10, mode='reflect'))
        images[im_nr] = im
        if gfm == None:
            gfm = images[im_nr].unsqueeze(-1)
        else:
            gfm = torch.cat((gfm, images[im_nr].unsqueeze(-1)), dim=-1)

    return gfm.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # S,C,H,W,L,1

def create_next_folder(path):
    folder = 0
    while True:
        if not os.path.exists(os.path.join(path)):
            os.mkdir(os.path.join(path))
        if not os.path.exists(os.path.join(path, f'{folder}')):
            os.mkdir(os.path.join(path, f'{folder}'))
            path = os.path.join(path, f'{folder}')
            break
        else:
            folder += 1
    return path

def get_real_samples(path):
    '''file_path = os.path.join(path, '**', f'*.jpg')
    paths = glob.glob(file_path, recursive=True)
    numbers = [x[-6:-4] for x in paths]'''

    folders = glob.glob(os.path.join(path, '*'), recursive=True)

    real_samples = None
    for folder in folders:
        file_path = os.path.join(folder, f'*.jpg')
        paths = glob.glob(file_path, recursive=True)
        numbers = [x[-6:-4] for x in paths]
        images = [None] * len(numbers)
        for idx, number in enumerate(numbers):
            if number[0] == '/':
                number = number[1]
            images[int(number)] = paths[idx]

        convert_tensor = transforms.ToTensor()

        samples = None
        for image in images:
            try:
                imageGrayscale = imageio.imread(image)
            except:
                pass
            im = convert_tensor(Image.fromarray(imageGrayscale))[0].unsqueeze(-1)
            if samples == None:
                samples = im
            else:
                samples = torch.cat((samples, im), dim=-1)
        if real_samples == None:
            real_samples = samples.unsqueeze(0)
        else:
            real_samples = torch.cat((real_samples, samples.unsqueeze(0)), dim=0)
    return real_samples.unsqueeze(1)

def get_scene_locations(batch_real_samples):
    resolution = (batch_real_samples, 386, 516)

    cameraDistance = 8.0679
    camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])

    h1, h2, h3 = 0.79, 3.55, 6.72178
    r1, r2, r3 = 2.975, 2.904641, 2.376113861

    locationLights = [[0.0, -r3, h3], [r3, 0.0, h3],
                      [0.0, r3, h3], [-r3, 0.0, h3],
                      [0.0, -r2, h2], [r2, 0.0, h2],
                      [0.0, r2, h2], [-r2, 0.0, h2],
                      [0.0, -r1, h1], [r1, 0.0, h1],
                      [0.0, r1, h1], [-r1, 0.0, h1]]
    lights = torch.tensor(locationLights)

    surface = torch.zeros(resolution)

    return camera, lights, surface

def get_scene_parameters(path):
    surface = torch.load(os.path.join(path, 'surface.pt'))
    lights = torch.load(os.path.join(path, 'lights.pt'))
    camera = torch.load(os.path.join(path, 'camera.pt'))

    rough = torch.load(os.path.join(path, 'rough.pt'))
    diffuse = torch.load(os.path.join(path, 'diffuse.pt'))
    reflectance = torch.load(os.path.join(path, 'reflectance.pt'))
    light_intensity = torch.load(os.path.join(path, 'light_intensity.pt'))
    intensity = torch.load(os.path.join(path, 'intensity.pt'))
    shadow = torch.load(os.path.join(path, 'shadow.pt'))

    x = torch.load(os.path.join(path, 'x.pt'))
    y = torch.load(os.path.join(path, 'y.pt'))

    return {'surface':surface, 'lights':lights, 'camera':camera,
            'rough':rough, 'diffuse':diffuse, 'reflectance':reflectance,
            'light_intensity':light_intensity, 'intensity':intensity,
            'shadow': shadow,
            'x':x, 'y':y}

class TimeDelta():
    def __init__(self):
        super().__init__()
        self.time = None
    def event(self):
        eventtime = datetime.now()
        if self.time != None:
            timedelta = eventtime - self.time
        else:
            timedelta = 0
        self.time = eventtime
        return timedelta


def get_height_profile(surface):

    B,H,W = surface.shape
    height_profile_x = surface.cpu().detach().numpy()[0, H//2, :]
    height_profile_y = surface.cpu().detach().numpy()[0, :, W//2]

    return height_profile_x, height_profile_y
