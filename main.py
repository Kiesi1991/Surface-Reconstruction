import os.path
from optimize_parameters import optimizeParameters
from mainFilament import train_NN
from utils import get_scene_parameters, get_light_attenuation

load_parameters = False
train = False

'''roughs = [(0.1,0.5,True),(0.2,0.5,True),(0.8,0.5,True), (0.9,0.5,True)]
diffuses = [(0.1,0.5,True),(0.2,0.5,True),(0.8,0.5,True), (0.9,0.5,True)]
reflactances = [(0.1,0.5,True),(0.2,0.5,True),(0.8,0.5,True), (0.9,0.5,True)]'''

roughs = [(0.2,0.2,True)]#, (0.5,0.5,True), (0.8,0.8,True)]
diffuses = [(0.2,0.2,True)]#, (0.5,0.5,True), (0.8,0.8,True)]
reflactances = [(0.2,0.2,True)]#, (0.5,0.5,True), (0.8,0.8,True)]

light_attenuation = get_light_attenuation()

for rough in roughs:
    for diffuse in diffuses:
        for reflactance in reflactances:

            if load_parameters:
                path = os.path.join('results', 'optimization', '1', 'Epoch-3000')
                print(f'1) load parameter from {path}')
                parameters = get_scene_parameters(path)
            else:
                print('1) optimize scene parameters for training')
                parameters = optimizeParameters(path_target='realSamples1', path_results=os.path.join('results', 'optimization', 'rough-diffuse-reflactance-test'),
                                                epochs=40001, intensity=5.0, weight_decay=0.0, light_attenuation=light_attenuation,
                                                # (synthetic, initial, parameter)
                                                rough=rough, diffuse=diffuse, reflectance=reflactance,
                                                synthetic=True, surface_opimization=False, quick_search=True, plot_every=1000)
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