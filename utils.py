from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import os
import glob
import imageio
from PIL import Image
from torchvision import transforms
from filament import evaluate_point_lights
import torch.nn.functional as tfunc

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
    '''
    scan folder with subfolders of real samples and create a dictionary
    :param path_real_samples: (string), directory, which should be scanned
    :return: (dictionary), keys: light source from 0 to 11, values: pytorch tensor (H, W, S), S = amount of cap-samples
    '''
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
    '''
    calculates gaussian filtered median (gfm) for samples in a folder
    :param path_real_samples: (string), directory, which should be scanned
    :return: (1, 1, H, W, L, 1), gaussian filtered median (gfm)
    '''
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

def createNextFolder(path):
    '''
    scans a given directory path and creates the next available folder. If "i" is already created, the function creates a folder named "i+1". "i" starts at 0 and increases by 1.
    :param path: (string), directory path to a folder
    :return: path + "i+1"
    '''
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

def getRealSamples(path):
    '''
    :param path: (string), directory path to real cabin cap samples, path to a folder with "S" subfolders.
    :return: (B, 1, H, W, L), pytorch tensor - data from cabin cap image samples.
    '''
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

def getScene(batch=1, H=386, W=516):
    '''
    returns a scene for a Freesurf sensor with a flat (zero height) surface matrix.
    :param batch: (int), batchsize - B flat surfaces will be created
    :return: (tuple) -> (cam, lights, surface),
    cam: (1, 1, 1, 1, 3), camera position in 3D space
    lights: (L,3), light positions in 3D space
    surface: (B, H, W), pytorch tensor with zeros as values -> flat surface
    '''
    resolution = (batch, H, W)
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

def getHeightProfile(surface, divide_by_mean = True):
    '''
    returns a height profile in x- and y-direction, given a surface matrix. If more than one samples (batches) available, the first sample is chosen.
    Profile in x-direction corresponds to the middle row of the surface matrix and the profile in y-direction is the middle column.
    :param surface: (B, H, W), surface matrix in pixel-to-height representation, every entry contains a height value in z-direction
    :param divide_by_mean: (boolean), if True return height profiles diveded by mean value.
    :return: (tuple) -> (height_profile_x, height_profile_y)
    height_profile_x: (W,), middle row numpy array of surface matrix
    height_profile_y: (H,), middle column numpy array of surface matrix
    '''
    B,H,W = surface.shape
    height_profile_x = surface.cpu().detach().numpy()[0, H//2, :]
    height_profile_y = surface.cpu().detach().numpy()[0, :, W//2]
    if divide_by_mean:
        return height_profile_x / height_profile_x.mean(), height_profile_y / height_profile_y.mean()
    else:
        return height_profile_x, height_profile_y

def getLightNumeration(level):
    '''
    :param level: (string), levels of light sources
    :return: (tuple) -> (start:int, end:int), values between "start" and "end" are selected light sources.
    '''
    if level=='level 1':
        return 8, 11
    elif level=='level 2':
        return 4, 7
    elif level=='level 3':
        return 0, 3
    elif level=='level 2+3':
        return 0, 7
    else:
        return 0, 11

def getOptimizedParameters(path):
    surface = torch.load(os.path.join(path, 'surface.pt'))
    lights = torch.load(os.path.join(path, 'lights.pt'))
    camera = torch.load(os.path.join(path, 'camera.pt'))
    rough = torch.load(os.path.join(path, 'rough.pt'))
    diffuse = torch.load(os.path.join(path, 'diffuse.pt'))
    reflectance = torch.load(os.path.join(path, 'reflectance.pt'))
    shadow = torch.load(os.path.join(path, 'shadow.pt'))

    return {'surface':surface, 'lights':lights, 'camera':camera,
            'rough':rough, 'diffuse':diffuse, 'reflectance':reflectance,
            'shadow': shadow}

def filament_renderer(surface, camera, lights,
                      rough=0.5, diffuse=0.5, reflectance=0.5):
    '''
    calculates intermediate variables for Filament renderer and apply Filament renderer
    :param surface: (B, H, W), surface matrix in pixel-to-height representation
    :param camera: (1, 1, 1, 1, 3), camera position
    :param lights: (12, 3), light positions
    :param rough: (int(pytorch Parameter)), material parameter
    :param diffuse: (int(pytorch Parameter)), material parameter
    :param reflectance: (int(pytorch Parameter)), material parameter
    :return: (1, 1, H, W, 12, 1), rendered output of Filament renderer
    '''
    lights = lights.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    light_dir = getVectors(surface, lights, norm=False).permute(0,2,3,1,4).unsqueeze(1)#B,1,H,W,L,3
    light_dir = tfunc.normalize(light_dir, dim=-1)

    roughness = (torch.ones((1, 1, 1, 1, 1, 1)).to(surface.device) * rough).to(surface.device)
    perceptual_roughness = roughness ** 0.5
    f0 = 0.16 * reflectance ** 2

    N = getNormals(surface)[:, :, :, :, None, :]  # 1,1,H,W,1,3
    V = getVectors(surface, camera).permute(0, 2, 3, 1, 4).unsqueeze(1)  # 1,1,H,W,1,3
    L_ = getVectors(surface, lights, norm=False).permute(0, 2, 3, 1, 4).unsqueeze(1) # 1,1,H,W,L,3

    light_attenuation = 1/(torch.linalg.norm(L_, axis=-1, keepdims=True)**2)
    L = normalize(L_)

    NoL = (N * L).sum(dim=-1, keepdim=True)
    NoV = (N * V).sum(dim=-1, keepdim=True)

    return evaluate_point_lights(
        light_color=torch.ones_like(torch.ones((1,1,1,1,12,1))).to(surface.device), # S?,C?,1,1,L?,CH
        light_intensity=torch.ones((1,1,1,1,12,1)).to(surface.device), # S?,C?,1,1,L?,1
        light_attenuation=light_attenuation, # S,C,H,W,L,1
        diffuseColor=(torch.ones((1, 1, 1, 1, 1, 1)).to(surface.device)*diffuse).to(surface.device), # S?,C?,H?,W?,1,CH
        perceptual_roughness=perceptual_roughness, # S?,C?,H?,W?,1,1
        roughness=roughness, # S?,C?,H?,W?,1,1
        f0=(torch.ones((1, 1, 1, 1, 1, 1)).to(surface.device) * f0).to(surface.device), # S?,C?,H?,W?,1,CH?
        light_dir=light_dir, # S,C,H,W,L,3
        NoL=NoL, # S,C,H,W,L,1
        view_dir=V, # S,C,H,W,1,3
        NoV=NoV, # S,C,H,W,1,1
        normal=N, # S,C,H,W,1,3
        # not used:
        use_fast_smith=False,
        use_energy_compensation=False,
        dfg_multiscatter=None
    )

def createSurface(resolution, sigmas = (2.5,5,10,15), p = 0.02, h = 0.01):
    surface = np.zeros(resolution)
    for sigma in sigmas:
        sigma_var = np.random.normal(0, sigma*0.4)
        p_var = np.clip(np.random.normal(0, p * 0.8), 0.00001, 0.05)
        surface1 = random_walk(size=resolution, p=p+p_var)
        surface1 = gaussian_filter(surface1, sigma=sigma+sigma_var, mode='reflect')
        surface1 /= surface1.max()
        #surface1 *= (sigma+sigma_var)
        surface += surface1

    #surface -= surface.min()
    surface /= surface.max()
    h_var = np.random.normal(0, h*0.2)
    surface *= (h+h_var)
    return (torch.from_numpy(surface) - torch.mean(torch.from_numpy(surface))).float()

def random_walk(size, p, I=3, l=1, h=50):
    '''
    perform random walk method for bulky hill expansions.
    :param size: (tuple) -> (H:int, W:int), size of output matrix
    :param p: (float), percentage of starting points based on amount of pixels (H*W)
    :param p: (int), number of loops (iterations)
    :param l: (int), minimum length of step size S
    :param h: (int), maximum length of step size S
    :return: (numpy array) -> (H:int, W:int) output matrix
    '''
    result = np.zeros(size)
    for _ in range(I):
        # Return random integers from `low` (inclusive) to `high` (exclusive) for step size
        S = np.random.randint(low=l, high=h, size=1)[0]
        # create starting points for random walk
        starting_points = np.random.choice(np.array([1.0, 0.0]), size=size, p=[p, 1.0 - p])
        # sample S actions: 1=right, 2=left, 3=up, 4=down
        actions = np.random.randint(4, size=S) + 1
        h, w = size
        surface = np.zeros((h+S*2, w+S*2))
        surface[S:-S, S:-S] = starting_points
        x, y = 0, 0
        for action in actions:
            if action == 1:
                x += 1 # right
            if action == 2:
                x -= 1 # left
            if action == 3:
                y -= 1 # up
            if action == 4:
                y += 1 # down
            surface[S+y:(h+S+y), S+x:(w+S+x)] += starting_points
        result += surface[S:-S, S:-S]
    return np.clip(result, 0.0, 1.0)
