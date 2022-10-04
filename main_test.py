import matplotlib.pyplot as plt

from optimize_parameters import optimizeParameters
import os
from utils import *
import matplotlib.pyplot as plt

# (synthetic, initial)
rough = 0.5
diffuse = 0.5
reflactance = 0.5
selected_lights = 'all levels'
path_results = os.path.join('results', 'test-optimization', '0')
its = 1000
para_lights = True

result = np.zeros((100,200))
for _ in range(3):
    step_size = np.random.randint(100, size=1)[0] + 1
    result += random_walk(size=(100,200), step_size=step_size, p=0.02, p_var=0)

plt.imshow(result)
plt.colorbar()
plt.show()

parameters = optimizeParameters(path_real_samples='realSamples1', path_results=path_results, para_lights=para_lights,
                                    iterations=its,
                                    # (synthetic, initial)
                                    rough=rough, diffuse=diffuse, reflectance=reflactance, selected_lights=selected_lights)

print('TheEnd')