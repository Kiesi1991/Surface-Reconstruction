from models import OptimizeParameters
import statistics
from utils import *
from tqdm import tqdm

def optimizeParameters(path_real_samples='realSamples', path_results=os.path.join('results', 'optimization'),
                       lr=1e-4,
                       iterations=50000, selected_lights='all levels', para_lights=True,
                       rough=0.5, diffuse=0.5, reflectance=0.5,
                       regularization_function='exp', lam = 0.000001):
    '''
    function, which perform optimization loop, apply plotting functions, initialize optimization model and store optimized parameters
    :param path_real_samples: (str), directory path to real samples, every subfolder consists of 12 real cabin-cap images
    :param path_results: (str), directory path, where results should be stored
    :param lr: (float), learning rate during optimization
    :param iterations: (int), amount to optimization loops
    :param selected_lights: (str) -> ['all levels', 'level 1', 'level 2', 'level 3', 'level 2+3'], define which light source levels are relevant for optimization
    :param para_lights: (boolean), if False light positions are constant, otherwise light positions are trainable
    :param rough: (float), material parameter
    :param diffuse: (float), material parameter
    :param reflectance: (float), material parameter
    :param regularization_function: (str) -> ['exp', 'square'], regularisation function for light positions
    :param lam: (float), value between 0 and 1 - hyperparameter for regularisation of light postions
    :return: (dictionary), dictionary with optmized parameters (material parameter, light positions)
    '''
    # create results path directory
    if not os.path.exists(os.path.join(path_results)):
        os.mkdir(os.path.join(path_results))
    # preperations before starting optimization
    plot_every = iterations // 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = getRealSamples(path_real_samples)
    start, end = getLightNumeration(selected_lights)
    camera, lights, surface = getSceneLocations(batch=samples.shape[0])
    path = createNextFolder(path_results)
    # define optimization model
    model = OptimizeParameters(surface, (lights, para_lights), camera,
                               rough=rough, diffuse=diffuse, reflectance=reflectance)
    # transfer model parameters to "device"
    model.to(device)
    # define loss function and optimizer
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    # calculate amount of active lights -> depends on selected lights
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
            # calculate errors between real cabin-cap images and predictions from Filament renderer + apply regularisation
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

                os.mkdir(os.path.join(path, f'iteration-{iteration}'))
                path2 = os.path.join(path, f'iteration-{iteration}')

                model.plotImageComparism(samples, pred, path2)
                model.plotDiagrams(plot_every, path)

                model.createParametersFile(path2, selected_lights)
                model.saveParameters(path2)

    return {'rough' : model.rough.detach().cpu(),
            'diffuse' : model.diffuse.detach().cpu(),
            'reflectance' : model.reflectance.detach().cpu(),
            'lights' : model.lights.detach().cpu(),
            }