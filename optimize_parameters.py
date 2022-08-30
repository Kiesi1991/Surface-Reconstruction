from models import OptimizeParameters
import statistics
from utils import *
from tqdm import tqdm

def optimizeParameters(path_target='realSamples', path_results=os.path.join('results', 'optimization'),
                       lr=1e-4,
                       iterations=3001,
                       intensity=2.5, selected_lights='all', para_lights=True,
                       rough=0.5, diffuse=0.5, reflectance=0.5,
                       regularization_function='exp', lam = 0.000001):

    plot_every = iterations // 8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #create results path directory
    if not os.path.exists(os.path.join(path_results)):
        os.mkdir(os.path.join(path_results))

    samples = getRealSamples(path_target)
    start, end = getLightNumeration(selected_lights)

    camera, lights, surface = getSceneLocations(batch=samples.shape[0])
    path = createNextFolder(path_results)

    model = OptimizeParameters(surface, (lights, para_lights), camera,
                               rough=rough, diffuse=diffuse, reflectance=reflectance,
                               intensity=intensity)

    model.to(device)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    active_lights = abs(end - start) + 1

    # optimiziation loop with gradient steps and plotting intermediate results
    for iteration in tqdm(range(iterations)):
            pred = model.forward()
            # apply regularisation to light positions
            distance = torch.linalg.norm(lights.to(device) - model.lights, axis=-1)
            if regularization_function == 'exp':
                distance_err = torch.exp(torch.sum(distance, dim=-1))/active_lights
            elif regularization_function == 'square':
                distance_err = (torch.sum(distance, dim=-1) ** 2) / active_lights
            else:
                raise(f'Regularisation function is not defined: {regularization_function}!')

            # set requires_grad value for light positions to False for the first iteration till iteration is grater equal to plot_every
            if iteration >= plot_every and para_lights:
                model.lights.requires_grad = True
            else:
                model.lights.requires_grad = False
            # calculate errors between real cabin-cap images and predictions from Filament renderer
            err = mse(pred[..., start:end+1].to(device), samples[..., start:end+1].to(device)) + lam * distance_err
            model.errs.append(err.item())
            # set gradients to zero and apply optimization step
            optimizer.zero_grad()
            err.backward()
            optimizer.step()


            if iteration % 50 == 0:
                model.errors.append(statistics.mean(model.errs[-10:]))

            # apply some plotting functions to the intermediate results
            if iteration % plot_every == 0:

                model.roughs.append(model.rough.cpu().detach().numpy().item())
                model.diffuses.append(model.diffuse.cpu().detach().numpy().item())
                model.reflectances.append(model.reflectance.cpu().detach().numpy().item())
                model.intensities.append(model.intensity.cpu().detach().numpy().item())

                os.mkdir(os.path.join(path, f'iteration-{iteration}'))
                path2 = os.path.join(path, f'iteration-{iteration}')

                model.plotImageComparism(samples, pred, path2)
                model.plotDiagrams(plot_every, path)

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