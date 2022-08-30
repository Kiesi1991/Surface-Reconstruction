from filament import evaluate_point_lights
import torch.nn.functional as tfunc
from utils import *


def filament_renderer(surface, camera, lights, light_intensity,
                      rough=0.5, diffuse=0.5, reflectance=0.5):
    '''
    calculates intermediate variables for Filament renderer and apply Filament renderer
    :param surface: (B, H, W), surface matrix in pixel-to-height representation
    :param camera: (1, 1, 1, 1, 3), camera position
    :param lights: (12, 3), light positions
    :param light_intensity: (1, 1, 1, 1, 12, 1), light intensities of all light sources
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
        light_color=torch.ones_like(light_intensity).to(surface.device), # S?,C?,1,1,L?,CH
        light_intensity=light_intensity.to(surface.device), # S?,C?,1,1,L?,1
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