import torch
import glob, os
import imageio
from PIL import Image
from torchvision import transforms
from models import OptimizeParameters
import matplotlib.pyplot as plt
import cv2
from utils import surface_height_distance


resolution = (386, 516)

cameraDistance = 8.
camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168


locationLights = [[0.0, -r3, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [-r3, 0.0, h3],
                  [0.0, -r2, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [-r2, 0.0, h2],
                  [0.0, -r1, h1], [r1, 0.0, h1],
                  [0.0, r1, h1],  [-r1, 0.0, h1]]

'''locationLights = [[ 0.5025, -2.4060,  1.3907],
        [ 2.3715, -0.2657,  1.3804],
        [-0.1163,  2.3329,  1.4274],
        [-2.2779,  0.3018,  1.4543],
        [ 0.3971, -2.4663,  3.6350],
        [ 2.4868, -0.2689,  3.4896],
        [-0.0831,  2.6326,  3.3804],
        [-2.6358,  0.6189,  3.2640],
        [ 0.2359, -2.4625,  5.0852],
        [ 2.5166, -0.1509,  5.1509],
        [ 0.2503,  2.2478,  5.2799],
        [-2.3585, -0.1042,  5.3759]]'''

lights = torch.tensor(locationLights)

file_path = os.path.join('realSamples', '**', f'*.jpg')
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
    im = convert_tensor(Image.fromarray(imageGrayscale))[0].unsqueeze(-1)
    if samples == None:
        samples = im
    else:
        samples = torch.cat((samples, im), dim=-1)

mesh = torch.zeros(resolution).unsqueeze(0)

model = OptimizeParameters(mesh, lights, camera)

color = model.forward()

from PIL import Image
import numpy as np
import statistics
'''
for idx in range(8):
    im = Image.fromarray(np.uint8(color[:,:,idx].detach().numpy()*255))
    im.show()
    #ims = Image.fromarray(np.uint8(samples[:, :, idx].detach().numpy() * 255))
    #ims.show()
    print(idx)'''

path = os.path.join('results')

folder = 0
while True:
    if not os.path.exists(os.path.join(path)):
        os.mkdir(os.path.join(path))
    if not os.path.exists(os.path.join(path, f'{folder}')):#, f'sigma={sigma},p={p}')):
        os.mkdir(os.path.join(path, f'{folder}'))#, f'sigma={sigma},p={p}'))
        path = os.path.join(path, f'{folder}')#, f'sigma={sigma},p={p}')
        break
    else:
        folder += 1

lr = 1e-3
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

epochs = 10001
errs = []
for epoch in range(epochs):
        pred = model.forward()
        surfaceDistance = surface_height_distance(model.mesh)
        err = mse(torch.exp(pred), torch.exp(samples)) #+ 1.0 * surfaceDistance
        errs.append(err.item())
        optimizer.zero_grad()
        err.backward()
        torch.nn.utils.clip_grad_value_(model.mesh, 0.001)
        optimizer.step()
        if epoch % 10 == 0:
            #im = Image.fromarray(np.uint8(pred[:, :, 5].detach().numpy() * 255))
            #im.show()
            print(f'Epoch {epoch} AVG Err {statistics.mean(errs[-10:])} Surface Max {model.mesh.detach().max()} Surface Min {model.mesh.detach().min()}')
        if epoch % 100 == 0:
            os.mkdir(os.path.join(path, f'Epoch-{epoch}'))
            path2 = os.path.join(path, f'Epoch-{epoch}')
            print(f'Rough {model.rough.item()} Diffuse {model.diffuse.item()} f0P {model.f0P.item()} Camera {model.camera.detach()}')
            num_L = samples.shape[2]
            for L in range(num_L):
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)

                plt.imshow(samples[:,:,L].cpu().detach().numpy())
                plt.clim(0, 1.0)

                plt.subplot(1, 2, 2)

                plt.imshow(pred[:,:,L].cpu().detach().numpy())
                plt.clim(0, 1.0)

                plt.savefig(os.path.join(path2, f'{L}.png'))
                plt.close()

                p = cv2.cvtColor(pred[:,:,L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
                t = cv2.cvtColor(samples[:,:,L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(t)
                plt.clim(0, 1.0)

                plt.subplot(1, 2, 2)
                plt.imshow(p)
                plt.clim(0, 1.0)

                plt.savefig(os.path.join(path2, f'TrueRGB-{L}.png'))
                plt.close()

            with open(os.path.join(path2, 'parameters.txt'), 'w') as f:
                f.write(f'Rough {model.rough.item()} Diffuse {model.diffuse.item()} f0P {model.f0P.item()} \n'
                        f'Camera {model.camera.detach()}\n'
                        f'Lights {model.lights.detach()}\n'
                        f'Surface {model.mesh.detach()}\n'
                        f'Surface Max {model.mesh.detach().max()}'
                        f'Surface min {model.mesh.detach().min()}\n'
                        f'Light Intensity {model.light_intensity.detach()}\n'
                        f'Light Color {model.light_color.detach()}\n'
                        f'X {model.x.detach()}\n'
                        f'Y {model.y.detach()}\n'
                        f'AVG Err {statistics.mean(errs[-10:])}')

            torch.save(model.mesh.detach(), os.path.join(path2, 'surface.pt'))

'''for idx in range(8):
    im = Image.fromarray(np.uint8(pred[:,:,idx].detach().numpy()*255))
    im.show()'''

print('TheEnd')