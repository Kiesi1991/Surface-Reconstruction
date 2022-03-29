import torch
import glob, os
import imageio
from PIL import Image
from torchvision import transforms
from models import OptimizeParameters


resolution = (386, 512)

cameraDistance = 8.0
camera = torch.tensor([[[[[0, 0, cameraDistance]]]]])

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[-r2, 0.0, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [0.0, -r2, h2],
                  [-r3, 0.0, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [0.0, -r3, h3]]

lights = torch.tensor(locationLights)

file_path = os.path.join('realSamples', '**', f'*.jpg')
images = glob.glob(file_path, recursive=True)

convert_tensor = transforms.ToTensor()

samples = None
for image in images:
    imageGrayscale = imageio.imread(image)
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
for idx in range(8):
    im = Image.fromarray(np.uint8(color[:,:,idx].detach().numpy()*255))
    im.show()

print('TheEnd')