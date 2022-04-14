„Meine
Ablage“ wird
mit
Verknüpfungen
übersichtlicher
gemacht …
Elemente, die
sich in mehr
als
einem
Ordner
befinden, werden in den
kommenden
Wochen
durch
Verknüpfungen
ersetzt.Der
Zugriff
auf
Dateien
und
Ordner
ändert
sich
nicht.Weitere
Informationen
from dataclasses import dataclass
import typing
import torch
import torch.nn.functional as tfunc
from cubes import get_cubes, load_exr
from mesh.mesh import Mesh
import nvdiffrast.torch as dr
from mesh.scene import Cameras, ImageLights, PointLights
from render.diffrast import DiffRastRenderer, DiffRastRendering
from render.filament import evaluate_ibl, evaluate_point_lights, get_distance_attenuation
from torchutil import atleast, take


@dataclass
class PbrRenderSettings:
    sum_lights: bool = False  # one image per light
    exposure: float = 1e-6
    ibl_correction: float = 10.
    with_specular: bool = True
    with_base_texture: bool = True
    with_reflectance_texture: bool = True
    with_roughness_texture: bool = True
    with_metallic_texture: bool = True
    with_normal_map: bool = True
    strict_normalize: bool = True
    use_fast_smith: bool = False
    use_energy_compensation: bool = True
    with_ibl = True
    cube_samples: int = 1024
    spherical_harmonics_bands: int = 3
    pos_gradient_boost: int = 3


def pbr_render(glctx,
               settings: PbrRenderSettings,
               cameras: Cameras,
               mesh: Mesh,
               point_lights: PointLights,
               cube_textures: torch.Tensor,  # list of S?,C?,6,HC',WC',L,CH
               spherical_harmonics: torch.Tensor,  # 9,S?,C?,L,CH
               dfg_multiscatter: torch.Tensor,  # W,W,2
               topology_hash
               ):
    with_texture = settings.with_base_texture or settings.with_metallic_texture \
                   or settings.with_reflectance_texture or settings.with_roughness_texture or settings.with_normal_map
    rendering = DiffRastRendering(glctx, cameras, mesh, with_texture)

    # Material
    base_color = rendering.get_matprop(mesh.base_color, settings.with_base_texture)  # S?,C?,H?,W?,1,CH
    reflectance = rendering.get_matprop(mesh.reflectance, settings.with_reflectance_texture)  # S?,C?,H?,W?,1,1
    perceptual_roughness = rendering.get_matprop(mesh.roughness, settings.with_roughness_texture)  # S?,C?,H?,W?,1,1
    metallic = rendering.get_matprop(mesh.metallic, settings.with_metallic_texture)  # S?,C?,H?,W?,1,1
    is_pure_metallic = metallic.numel() == 1 and metallic.item() == 1
    is_pure_dielectric = metallic.numel() == 1 and metallic.item() == 0
    perceptual_roughness = torch.clamp(perceptual_roughness, min=0.045)  # S?,C?,H?,W?,1,1|CH
    roughness = perceptual_roughness ** 2  # S?,C?,H?,W?,1,1|CH

    diffuseColor = None
    if not is_pure_metallic:
        dielectric = 1 - metallic  # S?,C?,H?,W?,1,1
        diffuseColor = dielectric * base_color  # S?,C?,H?,W?,1,CH

    f0 = 0
    if settings.with_specular:
        if not is_pure_metallic:
            f0 = 0.16 * reflectance ** 2 * dielectric  # S?,C?,H?,W?,1,1|CH
        if not is_pure_dielectric:
            f0 = f0 + base_color * metallic  # S?,C?,H?,W?,1,1|C

    # Geometry
    campos = torch.inverse(cameras.mv)[:, :, :3, 3]  # S,C,4,4 => S,C,3 camera positions
    vert_viewvec = campos[:, :, None, None, :] - mesh.vertices[:, None, :]  # S,C,V,1,3 View vectors at vertices

    if settings.with_normal_map and mesh.normal_map is not None:
        pix_normals = rendering.texture(mesh.normal_map / 2 + .5) * 2 - 1  # S,C,H,W,1,3
    else:
        pix_normals = rendering.interpolate(mesh.normals)  # S,C,H,W,1,3

    pix_normals = tfunc.normalize(pix_normals, dim=-1)  # S,C,H,W,1,3

    if settings.with_specular:
        pix_viewvec = rendering.interpolate(vert_viewvec)  # S,C,H,W,1,3
        if settings.strict_normalize:
            pix_viewvec = tfunc.normalize(pix_viewvec, dim=-1)  # S,C,H,W,1,3

        NoV = torch.sum(pix_normals * pix_viewvec, dim=-1, keepdim=True)  # S*C,H,W,1,1
    else:
        pix_viewvec = NoV = None

    color = None

    if point_lights is not None:
        vert_light = point_lights.position[:, :, None] - mesh.vertices[:, None,
                                                         :]  # S?,C?,V,L,3 from vertices to lights
        pix_light = rendering.interpolate(vert_light)  # S,C,H,W,L,3
        light_dir = tfunc.normalize(pix_light, dim=-1)  # S,C,H,W,L,3

        inv_falloff = 1 / point_lights.falloff[:, None, None, :, :]  # S?,C?,L?,1 => S?,C?,1,1,L?,1
        light_attenuation = get_distance_attenuation(pix_light, inv_falloff)  # S,C,H,W,L,1

        NoL = torch.sum(pix_normals * light_dir, dim=-1, keepdim=True)  # S,C,H,W,L,1

        intensity = point_lights.intensity[:, None, None, :, :]  # S?,C?,L?,1 => S?,C?,1,1,L?,1
        intensity = intensity * settings.exposure  # S?,C?,1,1,L?,1

        color = evaluate_point_lights(
            point_lights.color[:, :, None, None, :, :],  # S?,C?,L?,CH => S?,C?,1,1,L?,CH
            intensity,  # S?,C?,1,1,L?,1
            light_attenuation,
            diffuseColor,
            perceptual_roughness,
            roughness,
            f0,
            light_dir,
            NoL,
            pix_viewvec,
            NoV,
            pix_normals,
            settings.use_fast_smith,
            settings.use_energy_compensation,
            dfg_multiscatter,
            settings.with_specular
        )  # S,V,H,W,L,CH

    if settings.with_ibl and spherical_harmonics is not None:
        L = spherical_harmonics.shape[-2]  # 9,S?,C?,L,CH

        ibl_color = evaluate_ibl(
            diffuseColor,
            perceptual_roughness,
            NoV,
            f0,
            dfg_multiscatter,
            pix_normals,
            pix_viewvec,
            cube_textures,
            spherical_harmonics,
            settings.spherical_harmonics_bands,
            settings.with_specular
        )  # S,C,H,W,LC,CH

        color = ibl_color if color is None else torch.cat((color, ibl_color), dim=-2)  # S,C,H,W,L,CH

    if settings.sum_lights and color is not None:
        color = color.sum(dim=-2, keepdim=True)  # S,C,H,W,1,CH

    return rendering.antialias(color, topology_hash, settings.pos_gradient_boost)  # S,C,H,W,L,CH ; S,C,H,W,1,1


class PbrRenderer(DiffRastRenderer):

    def __init__(self, init_context=False, device='cuda'):
        super().__init__(init_context)
        self.settings = PbrRenderSettings()
        self._cubes = None

        dfg = load_exr('../data/dfg.exr', device)  # D,D,3
        self._dfg_multiscatter = torch.stack((dfg[:, :, 1], dfg.sum(dim=-1)), dim=-1).permute(2, 0, 1)  # 2,D,D

    def set_image_lights(self, image_lights: ImageLights):
        if image_lights is None or image_lights.equirect is None:
            self._cube_textures = None
            self._spherical_harmonics = None
            return

        S, C, L, H, _, CH = image_lights.equirect.shape  # S,C,L,H,2H,CH
        equirect = image_lights.equirect.reshape(S * C * L, H, 2 * H, CH)  # B=SCL,H,2H,CH
        cubes = get_cubes(equirect, './cache/cubes/exr', self.settings.cube_samples)

        tex = cubes.tex  # list of B=SCL,6,H',W',CH
        tex = [t.reshape(S, C, L, 6, *t.shape[2:4], CH) for t in cubes.tex]  # list of S,C,L,6,H',W',CH
        tex = [t.permute(0, 1, 3, 4, 5, 2, 6) for t in tex]  # list of S,C,6,HC',WC',L,CH

        sh = cubes.sh  # B=SCL,9,CH
        sh = sh.reshape(S, C, L, 9, CH).permute(3, 0, 1, 2, 4)  # 9,S,C,L,CH

        self._cube_textures = tex  # list of S,C,6,HC',WC',L,CH
        self._spherical_harmonics = sh  # 9,S,C,L,CH

    def render(self,
               cameras: Cameras,  # C==B
               mesh: Mesh = None,
               point_lights: PointLights = None,
               image_lights_ind: typing.Tuple = None,  # slices or indices for S,L,C; use set_cube_lights() before
               image_lights_intensity: torch.Tensor = None,  # S?,C?,L?,1
               ) -> torch.Tensor:  # S,C,H,W,L,CH ; S,C,H,W,1,1

        if mesh is None:
            mesh = self._mesh

        cube_textures = None
        spherical_harmonics = None

        if image_lights_ind is not None:
            # s,c,l = [index_to_slice(ind) for ind in image_lights_ind]
            s, c, l = image_lights_ind

            image_lights_intensity = image_lights_intensity * (self.settings.exposure * self.settings.ibl_correction)
            intensity = image_lights_intensity[:, :, None, None, None, :, :]  # S?,C?,1,1,1,L?,1

            if self.settings.with_specular:
                cube_textures = [take(t, (s, c, None, None, None, l)) for t in
                                 self._cube_textures]  # list of S,C,6,HC',WC',L,CH
                cube_textures = [t * intensity for t in cube_textures]

            # spherical_harmonics = self._spherical_harmonics[:,s][:,:,c][:,:,:,l] #9,S,C,L,CH
            spherical_harmonics = take(self._spherical_harmonics, (None, s, c, l))  # 9,S,C,L,CH
            spherical_harmonics = spherical_harmonics * image_lights_intensity[None, :, :, :, :]  # 9,S,C,L,CH

        return pbr_render(
            DiffRastRenderer.context(),
            self.settings,
            cameras,
            mesh,
            point_lights,
            cube_textures,
            spherical_harmonics,
            self._dfg_multiscatter,
            self._topology_hash,
        )