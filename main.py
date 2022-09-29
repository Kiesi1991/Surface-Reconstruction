import os.path

import torch
import glob
from torchvision import transforms
from PIL import Image
import imageio

from models import *

from optimize_parameters import optimizeParameters
from training import train_NN
from utils import get_scene_parameters, get_light_attenuation, get_height_profile
import matplotlib.pyplot as plt

path = os.path.join('results', 'optimization', '3', '0', 'Epoch-100000')

parameters = get_scene_parameters(path)

gt_surface = parameters['surface']#

model = ResidualNetwork()
model.load_state_dict(torch.load(os.path.join('results', 'trainNN', '3', '39', 'model.pt')))
model.eval()

file_path = os.path.join('realSamples1', 'part5', f'*.jpg')
paths = glob.glob(file_path, recursive=True)
numbers = [x[-6:-4] for x in paths]

images = [None]*len(numbers)
for idx, number in enumerate(numbers):
    if number[0] == '/':
        number = number[1]
    images[int(number)] = paths[idx]

convert_tensor = transforms.ToTensor()

samples = None
for image in images:
    try:
        imageGrayscale = imageio.imread(image)
    except:
        pass
    im = convert_tensor(Image.fromarray(imageGrayscale))[0].unsqueeze(0)
    if samples == None:
        samples = im
    else:
        samples = torch.cat((samples, im), dim=0)

pred_surface = model(samples.unsqueeze(0))

from scipy.ndimage import gaussian_filter
crop = 50
filter = 40

plt.figure(figsize=(30, 10))
plt.subplot(1, 3, 1)

gt_surface1 = gaussian_filter(gt_surface[0,crop:-crop,crop:-crop].cpu().detach().numpy(), sigma=20, mode='reflect')
gt_im = gt_surface[0,crop:-crop,crop:-crop].cpu().detach().numpy() - gt_surface1
plt.imshow(gt_im)
plt.title('ground truth')
plt.colorbar()
#plt.clim(0, 1.0)



plt.subplot(1, 3, 2)

pred_surface1 = gaussian_filter(pred_surface[0,crop:-crop,crop:-crop].cpu().detach().numpy(), sigma=20, mode='reflect')
pred_im = pred_surface[0,crop:-crop,crop:-crop].cpu().detach().numpy() - pred_surface1
plt.imshow(pred_im)
#plt.imshow(gt_surface1)
plt.title('prediction')
plt.colorbar()
#plt.clim(0, 1.0)

#plt.savefig(os.path.join(path2, f'{L}.png'))
#plt.close()

plt.subplot(1, 3, 3)

#filtered_difference = gaussian_filter(gt_im - pred_im, sigma=20, mode='reflect')
difference = (gt_im - pred_im)[filter:-filter,filter:-filter]

plt.imshow(difference)
#plt.imshow(filtered_difference)
plt.title('difference')
plt.colorbar()
#plt.clim(0, 1.0)

plt.show()

mse = (difference**2).mean()
print(mse)

print('2) train NN with fake images')
train_NN(camera=parameters['camera'],
         lights=parameters['lights'],
         light_intensity=parameters['light_intensity'],
         intensity=parameters['intensity'],
         rough=parameters['rough'],
         diffuse=parameters['diffuse'],
         reflectance=parameters['reflectance'],
         shadow=parameters['shadow'],
         x=parameters['x'],
         y=parameters['y'])