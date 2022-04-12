from shaders import FilamentShading
import torch
import os
import matplotlib.pyplot as plt
import cv2

path = os.path.join('results', '152', 'Epoch-2500')
surface = torch.load(os.path.join(path, 'surface.pt'))

rough = 0.49109870195388794
diffuse = 0#1.1094415187835693
f0P = 0.9966470003128052
camara = torch.tensor([[[[[-0.1162,  0.1845,  8.0112]]]]])
lights = torch.tensor([[ 0.2115, -2.7453,  1.0044],
        [ 2.7475, -0.2686,  1.0016],
        [ 0.2363,  2.7275,  1.0203],
        [-2.7223, -0.2796,  1.0253],
        [ 0.2062, -2.5007,  3.4476],
        [ 2.5474, -0.1986,  3.4049],
        [-0.1538,  2.6472,  3.3117],
        [-2.7820,  0.2326,  3.1705],
        [ 0.2201, -2.0781,  5.6605],
        [ 2.1423,  0.1658,  5.5819],
        [-0.2379,  2.1371,  5.5851],
        [-2.1259,  0.2258,  5.6304]])
x = 1.1103825569152832
y = 1.4844008684158325
shader = FilamentShading(camera=camara, lights=lights,rough=rough,diffuse=diffuse,f0P=f0P,x=x, y=y)

color = shader.forward(surface)

for im_nr in range(12):
    p = cv2.cvtColor(color[0,im_nr,:,:].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
    plt.imshow(p)
    plt.clim(0,1.0)
    plt.savefig(os.path.join(path, 'trial', f'{im_nr}-{diffuse}.png'))
    plt.close()
    #plt.show()
    #print(im_nr)

print('TheEnd')