import os.path
from optimize_parameters import optimizeParameters
from mainFilament import train_NN
from utils import get_scene_parameters

load_parameters = False

if load_parameters:
    path = os.path.join('results', 'optimization', '1', 'Epoch-3000')
    print(f'1) load parameter from {path}')
    parameters = get_scene_parameters(path)
else:
    print('1) optimize scene parameters for training')
    parameters = optimizeParameters(epochs=20001, intensity=2.5,
                                    rough=(0.1,0.1), diffuse=(0.1,0.1), f0P=(0.9,0.9),
                                    par_r=False, par_d=False, par_f0=False,
                                    synthetic=True)
print('---'*35)
'''print('2) train NN with fake images')
train_NN(camera=parameters['camera'],
         lights=parameters['lights'],
         light_intensity=parameters['light_intensity'],
         intensity=parameters['intensity'],
         rough=parameters['rough'],
         diffuse=parameters['diffuse'],
         f0P=parameters['f0P'],
         x=parameters['x'],
         y=parameters['y'])'''