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
from utils import get_light_attenuation, getVectors

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[0.0, -r3, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [-r3, 0.0, h3],
                  [0.0, -r2, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [-r2, 0.0, h2],
                  [0.0, -r1, h1], [r1, 0.0, h1],
                  [0.0, r1, h1],  [-r1, 0.0, h1]]

locationLights = [[ 0.4610, -2.3101,  5.4689],
        [ 2.0637, -0.1640,  5.8244],
        [ 0.0524,  2.2153,  5.6867],
        [-1.9160,  0.4533,  5.9731],
        [ 0.2925, -2.0323,  3.9218],
        [ 2.0486, -0.1973,  3.8636],
        [ 0.0061,  2.4624,  3.6414],
        [-2.2224,  0.4564,  3.7725],
        [ 0.1593, -2.2010,  1.5920],
        [ 2.2989,  0.0636,  1.4572],
        [-0.0257,  2.6306,  1.1837],
        [-2.0910,  0.2743,  1.6005]]


lights = torch.tensor(locationLights)

cameraDistance = 8.
camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])

camera = torch.tensor([[[[[-0.4919,  0.8713,  7.7930]]]]])


#################################
#     get real images           #
#################################

path = 'part9'
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
model.load_state_dict(torch.load(os.path.join('results', '223', '39', 'model.pt')))
model.eval()

surface = model.forward(samples.unsqueeze(0))

shader = FilamentShading(camera, lights)

pred_images = shader.forward(surface)

la = get_light_attenuation()
length_per_pixel = 1.608325 / 512

samples = samples[:,50:-50,50:-50]
pred_images = pred_images[0,:,50:-50,50:-50]
surface = surface[:,50:-50,50:-50].detach() # B=1,H,W
lights = lights.unsqueeze(1).unsqueeze(1).unsqueeze(0) # 1,L,1,1,3
x = (1.202888087 * 286) / 386
y = (1.608325 * 416) / 516
distance = getVectors(surface,lights, x, y, norm=False)
distance = torch.linalg.norm(distance, axis=4, keepdims=True)
la2 = 1/ (distance ** 2)

light_intensity = torch.tensor([[0.8111],
            [0.7329],
            [0.6446],
            [0.7719],
            [1.5840],
            [1.3656],
            [1.0796],
            [0.7442],
            [1.9819],
            [1.7916],
            [1.6281],
            [1.9292]])

for L in range(12):

    real = cv2.cvtColor(samples[L,:,:].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(real)
    plt.clim(0, 1.0)

    pred = cv2.cvtColor(pred_images[L,:,:].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.clim(0, 1.0)

    plt.savefig(os.path.join(path, f'real-vs-pred-{L}.png'))
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(la[0,0,50:-50,50:-50,L,0])
    #plt.clim(0, 1.0)

    plt.subplot(1, 2, 2)
    plt.imshow(la2[0,L,:,:,0])
    #plt.clim(0, 1.0)

    plt.savefig(os.path.join(path, f'la-{L}.png'))
    plt.close()

surface = surface.squeeze().cpu().detach().numpy()
plt.imshow(surface)
plt.savefig(os.path.join(path, f'surface.png'))
plt.close()

print('TheEnd')