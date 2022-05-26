import os.path

import torch

from optimize_parameters import optimizeParameters
from mainFilament import train_NN
from utils import get_scene_parameters, get_light_attenuation, get_height_profile
import matplotlib.pyplot as plt

load_parameters = False
train = False

'''surface_path = os.path.join('results', 'optimization', 'rough-diffuse-reflactance-surface-TrueSurface', '1', 'Epoch-50000', 'surface.pt')
surface = torch.load(surface_path)
surface = get_height_profile(surface)'''
'''surface_path = os.path.join('results', 'optimization', 'rough-diffuse-reflactance-surface-TrueSurface', '1', 'Epoch-50000', 'surface.pt')
surface = get_height_profile(surface_path)

surface_line = surface.cpu().detach().numpy()[0, :,200]

x = np.linspace(0, len(surface_line) - 1, len(surface_line))
plt.plot(x, surface_line, label='ground truth')
plt.xlabel('x')
plt.ylabel('height')
plt.legend()
plt.show()
#plt.close()'''


'''roughs = [(0.1,0.5,True),(0.2,0.5,True),(0.8,0.5,True), (0.9,0.5,True)]
diffuses = [(0.1,0.5,True),(0.2,0.5,True),(0.8,0.5,True), (0.9,0.5,True)]
reflactances = [(0.1,0.5,True),(0.2,0.5,True),(0.8,0.5,True), (0.9,0.5,True)]'''

roughs = [(0.2,0.4,True)]#, (0.5,0.4,True), (0.8,0.4,True)]
diffuses = [(0.2,0.4,True)]#, (0.5,0.4,True), (0.8,0.4,True)]
reflactances = [(0.2,0.4,True)]#, (0.5,0.4,True), (0.8,0.4,True)]
intensities = [2.5]

light_attenuation = get_light_attenuation()

'''for i in range(12):
    im = light_attenuation.cpu().detach().numpy()[0, 0,:,:,i,0]
    plt.imshow(im)
    plt.show()'''

for rough in roughs:
    for diffuse in diffuses:
        for reflactance in reflactances:
            for intensity in intensities:
                if load_parameters:
                    path = os.path.join('results', 'optimization', '1', 'Epoch-3000')
                    print(f'1) load parameter from {path}')
                    parameters = get_scene_parameters(path)
                else:
                    print('1) optimize scene parameters for training')
                    parameters = optimizeParameters(path_target='realSamples1', path_results=os.path.join('results', 'optimization', 'rough-diffuse-reflactance-surface-Test'),
                                                    epochs=20001, intensity=intensity, weight_decay=0.0, light_attenuation=light_attenuation,
                                                    # (synthetic, initial, parameter)
                                                    rough=rough, diffuse=diffuse, reflectance=reflactance,
                                                    synthetic=False, surface_opimization=True, quick_search=False, plot_every=2000)
                print('---'*35)

if train:
    print('2) train NN with fake images')
    train_NN(camera=parameters['camera'],
             lights=parameters['lights'],
             light_intensity=parameters['light_intensity'],
             intensity=parameters['intensity'],
             rough=parameters['rough'],
             diffuse=parameters['diffuse'],
             reflectance=parameters['reflectance'],
             x=parameters['x'],
             y=parameters['y'])