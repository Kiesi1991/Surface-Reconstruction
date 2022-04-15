import torch
import os
import matplotlib.pyplot as plt
import cv2
from shaders import FilamentShading
from models import ResidualNetwork
from torchvision import transforms
import glob
import imageio
from PIL import Image

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[0.0, -r3, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [-r3, 0.0, h3],
                  [0.0, -r2, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [-r2, 0.0, h2],
                  [0.0, -r1, h1], [r1, 0.0, h1],
                  [0.0, r1, h1],  [-r1, 0.0, h1]]


lights = torch.tensor(locationLights)

cameraDistance = 8.
camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])

path = os.path.join('results', '193', 'Epoch-10000')

#################################
#     get real images           #
#################################

path = 'part5'
file_path = os.path.join(path, '**', f'*.jpg')
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

############################

model = ResidualNetwork()
model.load_state_dict(torch.load(os.path.join('results', '215', '39', 'model.pt')))
model.eval()

surface = model.forward(samples.unsqueeze(0))

shader = FilamentShading(camera, lights)

pred_images = shader.forward(surface)


for L in range(12):

    real = cv2.cvtColor(samples[L,50:-50,50:-50].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(real)
    plt.clim(0, 1.0)

    pred = cv2.cvtColor(pred_images[0,L,50:-50,50:-50].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.clim(0, 1.0)

    plt.savefig(os.path.join(path, f'real-vs-pred-{L}.png'))
    plt.close()

surface = surface.squeeze()[50:-50,50:-50].cpu().detach().numpy()
plt.imshow(surface)
plt.savefig(os.path.join(path, f'surface.png'))
plt.close()

print('TheEnd')