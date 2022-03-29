import torch

from filament import evaluate_point_lights
import torch.nn.functional as tfunc
from utils import *

'''h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[-r1, 0.0, h1], [r1, 0.0, h1],
                  [0.0, r1, h1],  [0.0, -r1, h1],
                  [-r2, 0.0, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [0.0, -r2, h2],
                  [-r3, 0.0, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [0.0, -r3, h3]]

lights = torch.tensor(locationLights)

cameraDistance = 8.0
camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])'''

def filament_renderer(surface, camera, lights, rough=0.5, diffuse=0.5, f0P=0.5, light_intensity=torch.ones((1, 1, 1, 1, 1, 1)), light_color=torch.ones((1, 1, 1, 1, 1, 1))):
    surface = surface
    lights = lights.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    B, H, W = surface.shape
    L = lights.size()[1]
    x = y = 1  # ???
    light_dir = getVectors(surface, lights, x, y, norm=False).permute(0,2,3,1,4).unsqueeze(1)#B,1,H,W,L,3
    #la = 1/torch.linalg.norm(light_dir, axis=4, keepdims=True).reshape((1,1,H,W,12,1))
    la = torch.ones((B, 1, H, W, L, 1), device=surface.device)
    #light_dir = normalize(light_dir).reshape((1,1,H,W,12,3))
    light_dir = tfunc.normalize(light_dir, dim=-1)

    roughness = (torch.ones((1, 1, 1, 1, 1, 1)) * rough).to(surface.device)
    perceptual_roughness = roughness ** 0.5

    N = getNormals(surface, x=x, y=y)[:, :, :, :, None, :]  # 1,1,H,W,1,3
    V = getVectors(surface, camera, x=x, y=y).permute(0, 2, 3, 1, 4).unsqueeze(1)  # 1,1,H,W,1,3
    L = getVectors(surface, lights, x=x, y=y).permute(0, 2, 3, 1, 4).unsqueeze(1) # 1,1,H,W,L,3

    #N = getNormals(surface.unsqueeze(0), x=W, y=H)
    #V = getVectors(surface.unsqueeze(0), camera, x=W, y=H)
    #L = getVectors(surface.unsqueeze(0), lights, x=W, y=H)

    NoL = (N * L).sum(dim=-1, keepdim=True)
    NoV = (N * V).sum(dim=-1, keepdim=True)

    #NoL = torch.einsum('abcde, abcde -> abcd', N, L).reshape(1,1,H,W,12,1)
    #NoV = torch.einsum('abcde, abcde -> abcd', N, V).reshape(1,1,H,W,1,1)

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

'''surface = createSurface(resolution=(512, 512), para=[(5,0.05),
         (10,0.05),
         (20,0.05),
         (30,0.05),
         (40,0.05),
         (50,0.05)])

color = filament_renderer(surface, camera, lights)

from PIL import Image
for idx in range(12):
    im = Image.fromarray(np.uint8(color[0,0,:,:,idx,0]*255))
    im.show()

print('TheEnd')'''