from filament import evaluate_point_lights
import torch.nn.functional as tfunc
from utils import *


def filament_renderer(surface, camera, lights, rough=0.5, diffuse=0.5, f0P=0.5, light_intensity=torch.ones((1, 1, 1, 1, 1, 1)), light_color=torch.ones((1, 1, 1, 1, 1, 1)), x=1.202888, y=1.608325, la=None):
    surface = surface
    lights = lights.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    B, H, W = surface.shape
    L = lights.size()[1]
    light_dir = getVectors(surface, lights, x, y, norm=False).permute(0,2,3,1,4).unsqueeze(1)#B,1,H,W,L,3
    light_dir = tfunc.normalize(light_dir, dim=-1)

    roughness = (torch.ones((1, 1, 1, 1, 1, 1)) * rough).to(surface.device)
    perceptual_roughness = roughness ** 0.5

    N = getNormals(surface, x=x, y=y)[:, :, :, :, None, :]  # 1,1,H,W,1,3
    V = getVectors(surface, camera, x=x, y=y).permute(0, 2, 3, 1, 4).unsqueeze(1)  # 1,1,H,W,1,3
    L = getVectors(surface, lights, x=x, y=y).permute(0, 2, 3, 1, 4).unsqueeze(1) # 1,1,H,W,L,3


    NoL = (N * L).sum(dim=-1, keepdim=True)
    NoV = (N * V).sum(dim=-1, keepdim=True)

    return evaluate_point_lights(
        light_color=light_color.to(surface.device),         # S?,C?,1,1,L?,CH
        light_intensity=light_intensity.to(surface.device),     # S?,C?,1,1,L?,1
        light_attenuation=la,                               # S,C,H,W,L,1 # MK: 1/r
        diffuseColor=(torch.ones((1, 1, 1, 1, 1, 1))*diffuse).to(surface.device),        # S?,C?,H?,W?,1,CH #MK: color from surface
        perceptual_roughness=perceptual_roughness,          # S?,C?,H?,W?,1,1 #MK: roughness**2
        roughness=roughness,                                # S?,C?,H?,W?,1,1 #MK: Reflexionsparameter
        f0=(torch.ones((1, 1, 1, 1, 1, 1)) * f0P).to(surface.device),            # S?,C?,H?,W?,1,CH? # wird aus der Diffusecolor berechnet (siehe pbr_render function)
        light_dir=light_dir,                                # S,C,H,W,L,3
        NoL=NoL,                                            # S,C,H,W,L,1
        view_dir=V, #.reshape(1,1,H,W,1,3),                    # S,C,H,W,1,3
        NoV=NoV,                                            # S,C,H,W,1,1
        normal=N, #.reshape(1,1,H,W,1,3),                      # S,C,H,W,1,3
        # not used:
        use_fast_smith=False,
        use_energy_compensation=False,
        dfg_multiscatter=None
    )