from models import OptimizeParameters
import matplotlib.pyplot as plt
import cv2
import statistics
from utils import *
from tqdm import tqdm
from shaders import FilamentShading

def optimizeParameters(path_target='realSamples', path_results=os.path.join('results', 'optimization'),
                       lr=1e-4, weight_decay=0.,
                       iterations=3001, gfm=None,
                       intensity=2.5, selected_lights='all', para_lights=True,
                       rough=(0.5,0.5,True), diffuse=(0.5,0.5,True), reflectance=(0.5,0.5,True),
                       synthetic=False, surface_opimization=True,
                       regularization_function='exp'):

    plot_every = iterations//8
    lam = 0.000001

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(os.path.join(path_results)):
        os.mkdir(os.path.join(path_results))

    samples = getRealSamples(path_target)
    start, end = getLightNumeration(selected_lights)

    camera, lights, mesh = getSceneLocations(batch=1) if synthetic else getSceneLocations(batch=samples.shape[0])
    path = createNextFolder(path_results)

    if synthetic:
        light_intensity = torch.ones((1,1,1,1,12,1))
        shader = FilamentShading(camera, lights, light_intensity, intensity=torch.tensor(intensity),
                                 rough=torch.tensor(rough[0]), diffuse=torch.tensor(diffuse[0]), reflectance=torch.tensor(reflectance[0]), device=device)
        surface = torch.tensor(createSurface((386, 516)).tolist()).unsqueeze(0).to(device)
        samples = shader.forward(surface).permute((0,2,3,1)).unsqueeze(0)

    model = OptimizeParameters((mesh,True) if surface_opimization else (surface,False),
                               (lights,para_lights), (camera,False), device=device, gfm=gfm,
                               rough=rough[1], diffuse=diffuse[1], reflectance=reflectance[1],
                               par_r=rough[2], par_d=diffuse[2], par_ref=reflectance[2], intensity=intensity)

    model.lights.requires_grad = False

    parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(name)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    active_lights = abs(end - start) + 1

    for iteration in tqdm(range(iterations)):
            pred = model.forward()
            distance = torch.linalg.norm(lights.to(device) - model.lights, axis=-1)
            if regularization_function == 'exp':
                distance_err = torch.exp(torch.sum(distance, dim=-1))/active_lights
            elif regularization_function == 'square':
                distance_err = (torch.sum(distance, dim=-1) ** 2) / active_lights
            else:
                raise(f'Regularisation function is not defined: {regularization_function}!')

            err = mse(pred[...,start:end+1].to(device), samples[...,start:end+1].to(device)) + lam * distance_err
            model.errs.append(err.item())
            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if iteration == plot_every and para_lights:
                model.lights.requires_grad = True

            if iteration % 50 == 0:
                model.errors.append(statistics.mean(model.errs[-10:]))

            if iteration % plot_every == 0:

                model.roughs.append(model.rough.cpu().detach().numpy().item())
                model.diffuses.append(model.diffuse.cpu().detach().numpy().item())
                model.reflectances.append(model.reflectance.cpu().detach().numpy().item())
                model.intensities.append(model.intensity.cpu().detach().numpy().item())

                os.mkdir(os.path.join(path, f'iteration-{iteration}'))
                path2 = os.path.join(path, f'iteration-{iteration}')

                model.plotImageComparism(samples, pred, path2)
                model.plotDiagrams(model, plot_every, path, synthetic,
                                   rough_origin=rough[0], reflectance_origin=reflectance[0], diffuse_origin=diffuse[0])

                model.createParametersFile(path2, selected_lights)
                model.saveParameters(path2)

    return {'rough' : model.rough.detach().cpu(),
            'diffuse' : model.diffuse.detach().cpu(),
            'reflectance' : model.reflectance.detach().cpu(),
            'camera' : model.camera.detach().cpu(),
            'lights' : model.lights.detach().cpu(),
            'light_intensity': model.light_intensity.detach().cpu(),
            'intensity': model.intensity.detach().cpu()
            }