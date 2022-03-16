# translated from https://github.com/google/filament GLSL to pytorch

import torch
import torch.nn.functional as tfunc
#from cubes import Cubes # not relevant for IBL
#import nvdiffrast.torch as dr #nvidearenderer, not relevant
# from torchutil import atleast # not relevant


# from filament brdf.fs/D_GGX()
def d_ggx(
        roughness: torch.Tensor,  # broadcast
        NoH: torch.Tensor,  # broadcast
):
    a = NoH * roughness
    k = roughness / (1.0 - NoH * NoH + a * a)
    return k * k * (1.0 / torch.pi)  # broadcast


# from filament brdf.fs/V_SmithGGXCorrelated()
def v_smith_ggx_correlated(
        roughness,  # broadcast
        NoV: torch.Tensor,  # broadcast
        NoL: torch.Tensor,  # broadcast
        eps=1e-8
):
    a2 = roughness * roughness
    lambdaV = NoL * torch.sqrt((NoV - a2 * NoV) * NoV + a2)
    lambdaL = NoV * torch.sqrt((NoL - a2 * NoL) * NoL + a2)
    v = 0.5 / (lambdaV + lambdaL + eps)
    return v  # broadcast


# from filament brdf.fs/V_SmithGGXCorrelated_Fast()
def v_smith_ggx_correlated_fast(
        roughness,  # broadcast
        NoV: torch.Tensor,  # broadcast
        NoL: torch.Tensor,  # broadcast
):
    return 0.5 / torch.lerp(2.0 * NoL * NoV, NoL + NoV, roughness)  # broadcast


# from filament brdf.fs/F_Schlick()
def f_schlick(
        f0: torch.Tensor,  # broadcast
        f90: torch.Tensor,  # broadcast
        VoH: torch.Tensor  # broadcast
):
    return f0 + (f90 - f0) * (1.0 - VoH) ** 5  # broadcast


# from filament brdf.fs/F_Schlick()
def f_schlick_1(
        f0: torch.Tensor,  # broadcast
        VoH: torch.Tensor  # broadcast
):
    f = (1.0 - VoH) ** 5
    return f + f0 * (1.0 - f)  # broadcast


# from filament brdf.fs/fresnel()
def fresnel(
        f0: torch.Tensor,  # broadcast
        LoH: torch.Tensor  # broadcast
):
    f90 = (f0.sum(dim=-1, keepdim=True) * (50.0 * 0.33)).clamp(max=1)
    return f_schlick(f0, f90, LoH)  # broadcast


# from filament brdf.fs/Fd_Lambert()
def fd_lambert():
    return 1.0 / torch.pi


# from filament brdf.fs/Fd_Burley()
def fd_burley(roughness, NoV, NoL, LoH):
    f90 = 0.5 + 2.0 * roughness * LoH * LoH
    lightScatter = f_schlick(1.0, f90, NoL)
    viewScatter = f_schlick(1.0, f90, NoV)
    return lightScatter * viewScatter * (1.0 / torch.pi)


# from filament shading_model_standard.fs/specularLobe()
def specular_lobe(
        roughness: torch.Tensor,  # broadcast
        f0: torch.Tensor,  # broadcast
        NoV: torch.Tensor,  # broadcast
        NoL: torch.Tensor,  # broadcast
        NoH: torch.Tensor,  # broadcast
        LoH: torch.Tensor,  # broadcast
        use_fast_smith=False
):
    # isotropic_lobe
    D = d_ggx(roughness, NoH)
    ggx = v_smith_ggx_correlated_fast if use_fast_smith else v_smith_ggx_correlated
    V = ggx(roughness, NoV, NoL)
    F = fresnel(f0, LoH)
    return D * V * F  # broadcast


# from filament shading_model_standard.fs/diffuseLobe()
def diffuse_lobe(
        diffuseColor: torch.Tensor,  # broadcast
        roughness: torch.Tensor,  # broadcast
        NoV: torch.Tensor,  # broadcast
        NoL: torch.Tensor,  # broadcast
        LoH: torch.Tensor,  # broadcast
):
    if True:
        d = diffuseColor * fd_lambert()
    else:
        d = diffuseColor * fd_burley(roughness, NoV, NoL, LoH)
    return d  # broadcast


# from filament light_indirect.fs/PrefilteredDFG_LUT()
def lookup_dfg(
        perceptual_roughness: torch.Tensor,  # S?,C?,H?,W?,1,1
        NoV: torch.Tensor,  # S,C,H,W,1,1
        dfg_multiscatter: torch.Tensor,  # 1|2,D,D
) -> torch.Tensor:  # S,C,1|2,H,W

    S, C, H, W, _, _ = NoV.shape

    D = dfg_multiscatter.shape[-1]
    input = dfg_multiscatter[None].expand(S * C, -1, -1, -1)  # S*C,1|2,D,D

    x = NoV.reshape(S * C, H, W, 1)
    y = 1 - perceptual_roughness.expand(S, C, H, W, 1, 1).reshape(S * C, H, W, 1)
    grid = torch.concat((x, y), dim=-1)  # S*C,H,W,2
    grid = grid * 2 - 1  # [0,1]=>[-1,1]

    return tfunc.grid_sample(input, grid, align_corners=False, padding_mode='border').reshape(S, C, -1, H,
                                                                                              W)  # S,C,1|2,H,W


# from filament shading_lit.fs/getEnergyCompensationPixelParams()
def energy_compensation(
        perceptual_roughness: torch.Tensor,  # S?,C?,H?,W?,1,1
        NoV: torch.Tensor,  # S,C,H,W,1,1
        dfg_multiscatter: torch.Tensor,  # 2,D,D
        f0: torch.Tensor,  # S?,C?,H?,W?,1,CH?
) -> torch.Tensor:  # S,C,H,W,CH?

    # TODO combine with IBL
    dfg2 = lookup_dfg(perceptual_roughness, NoV, dfg_multiscatter[[1]])  # S,C,1|2,H,W
    dfg2 = dfg2[:, :, 0, :, :, None, None]  # S,C,H,W,1,1
    return 1.0 + f0 * (1.0 / dfg2 - 1.0)  # S,C,H,W,1,CH?


# from filament shading_model_standard.fs/surfaceShading()
def evaluate_point_lights(
        light_color: torch.Tensor,  # S?,C?,1,1,L?,CH
        light_intensity: torch.Tensor,  # S?,C?,1,1,L?,1
        light_attenuation: torch.Tensor,  # S,C,H,W,L,1 # MK: 1/r
        diffuseColor: torch.Tensor,  # S?,C?,H?,W?,1,CH #MK: color from surface
        perceptual_roughness: torch.Tensor,  # S?,C?,H?,W?,1,1 #MK: roughness**2
        roughness: torch.Tensor,  # S?,C?,H?,W?,1,1 #MK: Reflexionsparameter
        f0: torch.Tensor,  # S?,C?,H?,W?,1,CH? # wird aus der Diffusecolor berechnet (siehe pbr_render function)
        light_dir: torch.Tensor,  # S,C,H,W,L,3
        NoL: torch.Tensor,  # S,C,H,W,L,1
        view_dir: torch.Tensor,  # S,C,H,W,1,3
        NoV: torch.Tensor,  # S,C,H,W,1,1
        normal: torch.Tensor,  # S,C,H,W,1,3
        use_fast_smith: bool, # MK: False (schneller rendern)
        use_energy_compensation: bool, # MK: False (Korrekturfaktor)
        dfg_multiscatter: torch.Tensor,  # 2,D,D # MK: (dann braucht man diesen Wert nicht), D = Größe der Lookuptabelle
        with_specular: bool = True, # MK: True (spiegelden Reflexionen)
) -> torch.Tensor:
    S, C, H, W, L, _ = NoL.shape
    h = tfunc.normalize(view_dir + light_dir, dim=-1)  # S,C,H,W,L,3

    NoL = torch.relu(NoL)  # S,C,H,W,L,1
    NoH = torch.relu(torch.sum(normal * h, dim=-1, keepdim=True))  # S,C,H,W,L,1
    LoH = torch.relu(torch.sum(light_dir * h, dim=-1, keepdim=True))  # S,C,H,W,L,1

    color = None

    if with_specular:
        Fr = specular_lobe(roughness, f0, NoV, NoL, NoH, LoH, use_fast_smith)  # S,C,H,W,L,CH?
        if use_energy_compensation:
            Fr = Fr * energy_compensation(perceptual_roughness, NoV, dfg_multiscatter, f0)  # S,C,H,W,1,CH?
        color = Fr  # S,C,H,W,1,CH?

    if diffuseColor is not None:  # None for pure metals
        Fd = diffuse_lobe(diffuseColor, roughness, NoV, NoL, LoH)  # S,C,H,W,L,CH
        color = color + Fd if color is not None else Fd  # S,C,H,W,L,CH

    if color is not None:
        colored_intensity = light_color * light_intensity  # S?,C?,1,1,L?,CH
        color = color * colored_intensity  # S,C,H,W,L,CH
        color = color * light_attenuation * NoL  # S,C,H,W,L,CH

    return color  # S,C,H,W,L,CH


# from filament light_punctual.fs/getSquareFalloffAttenuation()
def get_square_falloff_attenuation(
        distanceSquare: torch.Tensor,  # broadcast
        inv_falloff: torch.Tensor  # broadcast
):
    factor = distanceSquare * inv_falloff
    smoothFactor = torch.relu(1.0 - factor * factor)
    return smoothFactor * smoothFactor  # broadcast


# from filament light_punctual.fs/getDistanceAttenuation()
def get_distance_attenuation(
        posToLight: torch.Tensor,  # S,C,H,W,L,3
        inv_falloff: torch.Tensor  # S?,1,1,1,L?,1
):
    distanceSquare = (posToLight ** 2).sum(dim=-1, keepdim=True)  # S,C,H,W,L,1
    attenuation = get_square_falloff_attenuation(distanceSquare, inv_falloff)  # S,C,H,W,L,1
    # Assume a punctual light occupies a volume of 1cm to avoid a division by 0
    return attenuation / torch.clamp(distanceSquare, min=1e-4)  # S,C,H,W,L,1


# from filament light_indirect.fs(specularDFG()
def specular_dfg(
        perceptual_roughness: torch.Tensor,  # S?,C?,H?,W?,1,1
        NoV: torch.Tensor,  # S,C,H,W,1,1
        f0: torch.Tensor,  # S?,C?,H?,W?,1,CH?
        dfg_multiscatter: torch.Tensor,  # 2,D,D
) -> torch.Tensor:  # S,C,H,W,1,CH?

    dfg = lookup_dfg(perceptual_roughness, NoV, dfg_multiscatter)  # S,C,1|2,H,W
    dfg0 = dfg[:, :, 0, :, :, None, None]  # S,C,H,W,1,1
    dfg1 = dfg[:, :, 1, :, :, None, None]  # S,C,H,W,1,1
    return torch.lerp(dfg0, dfg1, f0)  # S,C,H,W,1,CH?


# from filament light_indirect.fs/getReflectedVector()
def get_reflected(
        pix_normals: torch.Tensor,  # S,C,H,W,1,3
        pix_viewvec: torch.Tensor,  # S,C,H,W,1,3
) -> torch.Tensor:  # S,C,H,W,1,3

    refl = pix_viewvec - 2.0 * pix_normals * torch.sum(pix_normals * pix_viewvec, -1, keepdim=True)  # S,C,H,W,1,3

    # TODO normalize required for cubemap lookup??
    refl = tfunc.normalize(refl, dim=-1)  # S,C,H,W,1,3

    return refl


# from filament light_indirect.fs/prefilteredRadiance()
'''def prefiltered_radiance(
        perceptual_roughness: torch.Tensor,  # S?,C?,H?,W?,1,1
        refl_vec: torch.Tensor,  # S,C,H,W,1,3
        cube_textures: torch.Tensor,  # list of S?,C?,6,HC',WC',L,CH
) -> torch.Tensor:  # S,C,H,W,L,CH

    S, C, H, W, _, _ = refl_vec.shape
    SC, CC, _, HC, WC, L, CH = cube_textures[0].shape

    if SC * CC > 1:
        cube_textures = [t.expand(S, C, -1, -1, -1, -1, -1).reshape(S * C, 6, *t.shape[3:5], L * CH) for t in
                         cube_textures]  # list of S*C,6,HC',WC',L*CH
    else:
        # can broadcast in dr.texture
        cube_textures = [t.reshape(1, 6, *t.shape[3:5], L * CH) for t in cube_textures]  # list of 1,6,HC',WC',L*CH

    tex = cube_textures[0]  # S*C?,6,HC,WC,L*CH
    mip = cube_textures[1:]  # list of S*C?,6,HC',WC',L*CH

    uv = -refl_vec.reshape(S * C, H, W, 3)  # S*C,H,W,1,3

    roughness_one_level = len(mip)
    lod = roughness_one_level * perceptual_roughness * (2.0 - perceptual_roughness)  # S?,C?,H?,W?,1,1
    SR, CR = lod.shape[:2]
    if SR * CR > 1:
        # not broadcastable in dr.texture
        lod = lod.expand(S, C, -1, -1, 1, 1)  # S,C,H?,W?,1,1
    lod = lod.expand(-1, -1, H, W, 1, 1).reshape(-1, H, W).contiguous()  # S*C?,H,W

    # dr limitations, TODO dr issue to support more broadcasting
    lod = lod.expand(S * C, -1, -1).contiguous()
    tex = tex.contiguous()
    uv = uv.contiguous()
    mip = [m.contiguous() for m in mip]

    radiance = dr.texture(tex=tex, uv=uv, mip_level_bias=lod, mip=mip, boundary_mode='cube')  # S*C,H,W,L*C
    return radiance.reshape(S, C, H, W, L, CH)'''


# from filament light_indirect.fs/Irradiance_SphericalHarmonics
def irradiance_spherical_harmonics(
        pix_normals: torch.Tensor,  # S,C,H,W,1,3
        spherical_harmonics: torch.Tensor,  # 9,S?,C?,L,CH
        spherical_harmonics_bands: int = 3,
) -> torch.Tensor:  # S,C,H,W,L,CH

    n = pix_normals
    nx = n[..., 0, None]  # S,C,H,W,1,1
    ny = n[..., 1, None]  # S,C,H,W,1,1
    nz = n[..., 2, None]  # S,C,H,W,1,1
    sh = spherical_harmonics[:, :, :, None, None, :, :]  # 9,S?,C?,1,1,L,CH

    d = sh[0] \
        + sh[1] * ny \
        + sh[2] * nz \
        + sh[3] * nx

    if spherical_harmonics_bands > 2:
        d = d \
            + sh[4] * (ny * nx) \
            + sh[5] * (ny * nz) \
            + sh[6] * (3 * nz ** 2 - 1) \
            + sh[7] * (nz * nx) \
            + sh[8] * (nx ** 2 - ny ** 2)

    d = d.clamp(min=0)
    return d  # S,C,H,W,L,CH


# from filament light_indirect.fs/evaluateIBL()
def evaluate_ibl(
        diffuseColor: torch.Tensor,  # S?,C?,H?,W?,1,CH
        perceptual_roughness: torch.Tensor,  # S?,C?,H?,W?,1,1
        NoV: torch.Tensor,  # S,C,H,W,1,1
        f0: torch.Tensor,  # S?,C?,H?,W?,1,CH?
        dfg_multiscatter: torch.Tensor,  # 2,D,D
        pix_normals: torch.Tensor,  # S,C,H,W,1,3
        pix_viewvec: torch.Tensor,  # S,C,H,W,1,3
        cube_textures: torch.Tensor,  # list of S?,C?,6,HC',WC',L,CH
        spherical_harmonics: torch.Tensor,  # 9,S?,C?,L,CH
        spherical_harmonics_bands: int = 3,
        with_specular: bool = True
):
    color = None

    if with_specular:
        spec_dfg = specular_dfg(perceptual_roughness, NoV, f0, dfg_multiscatter)  # S,C,H,W,1,CH?
        refl_vec = get_reflected(pix_normals, pix_viewvec)  # S,C,H,W,1,3
        radiance = prefiltered_radiance(perceptual_roughness, refl_vec, cube_textures)  # S,C,H,W,L,CH
        Fr = spec_dfg * radiance  # S,C,H,W,L,CH
        color = Fr  # S,C,H,W,L,CH

    # diffuse
    if diffuseColor is not None:  # None for pure metals
        diffuse_irradiance = irradiance_spherical_harmonics(pix_normals, spherical_harmonics,
                                                            spherical_harmonics_bands)  # S,C,H,W,L,CH
        # diffuseBRDF = 1 #no ambient occlusion
        Fd = diffuseColor * diffuse_irradiance  # S,C,H,W,L,CH
        color = color + Fd * (1.0 - spec_dfg) if color is not None else Fd  # S,C,H,W,L,CH

    return color