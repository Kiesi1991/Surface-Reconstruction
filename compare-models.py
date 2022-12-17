from models import *
from pathlib import Path
import torch
from shaders import FilamentShading
from statistics import mean

# importing library
import matplotlib.pyplot as plt

# by running this file, the trained models in "path_results" are compared with the mean synthetic gradient (MSE)

path_results = os.path.join('results', 'trainNN19')
file_path = os.path.join(path_results, f'*.pt')
paths = glob.glob(file_path, recursive=True)

model_parameter = {'model2.pt': {'L':16, 'CM':64, 'c':1, 'BlockNet':ResNextBlock},
                   'model20.pt': {'L':16, 'CM':64, 'c':1, 'BlockNet':ResNextBlock},
                   'model21.pt': {'L':16, 'CM':64, 'c':1, 'BlockNet':ResNextBlock},
                   'model22.pt': {'L':16, 'CM':64, 'c':1, 'BlockNet':ResNextBlock},
                   'model8.pt': {'L':8, 'CM':64, 'c':1, 'BlockNet':ConvBlock},}

path_real_samples = 'realSamples1'
real_samples = getRealSamples(path_real_samples).permute(0,4,2,3,1).squeeze(-1)

resolution = (386, 516)

path_optimized_parameters = os.path.join('results', 'optimization', '1', '0')
optimized_parameters = getOptimizedParameters(path_optimized_parameters)
transformation = FilamentShading(optimized_parameters)

mse = torch.nn.MSELoss() # MSE loss function

synthetic_surfaces = []
for _ in range(50):
    synthetic_surfaces.append(torch.tensor(createSurface(resolution).tolist()).unsqueeze(0))

x = []
y = []

for model_path in paths:
    error = []
    child = Path(model_path).absolute().parts[-1]
    model_nr = '.'.join(str(e) for e in child if e.isdigit())
    model = SurfaceNet(layers=model_parameter[child]['L'], mid_channels=model_parameter[child]['CM'], BlockNet=model_parameter[child]['BlockNet'], cardinality=model_parameter[child]['c'])
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')
    model.eval()
    for synthetic_surface in synthetic_surfaces:
        synthetic_image = transformation.forward(synthetic_surface.to('cuda'))
        predicted_surface = model(synthetic_image).to('cpu')
        synthetic_gradients = getGradients(synthetic_surface)
        predicted_gradients = getGradients(predicted_surface)
        error.append(mse(synthetic_gradients, predicted_gradients).item())
    print(f'{child}: {mean(error)}')
    x.append(f'model {model_nr}')
    y.append(float(format(mean(error), '.5f')))


# function to add value labels
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] / 2, y[i], ha='center')


# setting figure size by using figure() function
plt.figure(figsize=(10, 5))

# making the bar chart on the data
plt.bar(x, y)

# calling the function to add value labels
addlabels(x, y)

# giving Y labels
plt.ylabel("mean gradient error (MSE)")

# save the plot
plt.savefig(os.path.join(path_results, f'mean-gradient-error.png'))
plt.close()

print('TheEnd')