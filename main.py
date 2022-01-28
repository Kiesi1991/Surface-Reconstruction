import torch

from utils import *
import numpy as np
from shaders import PhongShading
from dataset import DummySet
from PIL import Image

# surface properties
length = 4
width = 2

resolution = (512, 1028)

surface = createSurface(resolution)

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[-r1, 0.0, h1], [r1, 0.0, h1],
                  [0.0, r1, h1],  [0.0, -r1, h1],
                  [-r2, 0.0, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [0.0, -r2, h2],
                  [-r3, 0.0, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [0.0, -r3, h3]]

cameraDistance = 8.0

dataset = DummySet(resolution)

shader = PhongShading(camera=[0, 0, cameraDistance], lights=locationLights, length=length, width=width)
I = shader.forward(dataset.data[0])



for idx, i in enumerate(I):
    im = Image.fromarray(np.uint8(i*255))
    im.show()

print('TheEnd')