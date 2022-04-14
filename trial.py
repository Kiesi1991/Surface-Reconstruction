from shaders import FilamentShading
import torch
import os
import matplotlib.pyplot as plt
import cv2

path = os.path.join('results', '193', 'Epoch-10000')
surface = torch.load(os.path.join(path, 'surface.pt'))

rough = 1.
diffuse = 0.70
f0P = 1.49
cameraDistance = 8.
camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[0.0, -r1, h1], [r1, 0.0, h1],
                  [0.0, r1, h1],  [-r1, 0.0, h1],
                  [0.0, -r2, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [-r2, 0.0, h2],
                  [0.0, -r3, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [-r3, 0.0, h3]]
lights = torch.tensor(locationLights)
x = torch.tensor(1.202888087)
y = torch.tensor(1.608325)
shader = FilamentShading(camera=camera, lights=lights,rough=rough,diffuse=diffuse,f0P=f0P,x=x, y=y)

color = shader.forward(surface)

path = os.path.join(path, 'trial')
if not os.path.exists(os.path.join(path)):
    os.mkdir(os.path.join(path))
path = os.path.join(path, f'r{rough}d{diffuse}f{f0P}')
if not os.path.exists(os.path.join(path)):
    os.mkdir(os.path.join(path))

for im_nr in range(12):
    p = cv2.cvtColor(color[0,im_nr,:,:].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
    plt.imshow(p)
    plt.clim(0,1.0)
    plt.savefig(os.path.join(path, f'{im_nr}.png'))
    plt.close()
    #plt.show()
    #print(im_nr)

print('TheEnd')