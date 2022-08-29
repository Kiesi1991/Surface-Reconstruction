from optimize_parameters import optimizeParameters
import os
from utils import *

rough = (0.2, 0.5, True)
diffuse = (0.2, 0.5, True)
reflactance = (0.2, 0.5, True)
intensity = 75
selected_lights = 'all levels'
path_results = os.path.join('results', 'test-optimization', '0')
its = 10000
para_lights = True
synthetic = False
mean_intensity = getGfm()

parameters = optimizeParameters(path_target='realSamples1', path_results=path_results, para_lights=para_lights,
                                    iterations=its, intensity=intensity, weight_decay=0.0, mean_intensity=mean_intensity,
                                    # (synthetic, initial, parameter)
                                    rough=rough, diffuse=diffuse, reflectance=reflactance, selected_lights=selected_lights,
                                    synthetic=synthetic, surface_opimization=True)

print('TheEnd')