import torch
import glob, os
import imageio
from PIL import Image
from torchvision import transforms
from models import OptimizeParameters


resolution = (386, 516)

cameraDistance = 7.5937
camera = torch.tensor([[[[[-0.2113, 0.3543, cameraDistance]]]]])

h1, h2, h3 = 0.79, 3.29, 5.79
r1, r2, r3 = 2.975, 2.660705446, 1.937933168

locationLights = [[-r2, 0.0, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [0.0, -r2, h2],
                  [-r3, 0.0, h3], [r3, 0.0, h3],
                  [0.0, r3, h3],  [0.0, -r3, h3]]

locationLights = [[0.0, -r2, h2], [r2, 0.0, h2],
                  [0.0, r2, h2],  [-r2, 0.0, h2],
                  [0.0, -r3, h3], [r3, 0.0, h3],
                  [0.0, r3, h3]]#,  [-r3, 0.0, h3]]

lights = torch.tensor(locationLights)

file_path = os.path.join('realSamples', '**', f'*.jpg')
paths = glob.glob(file_path, recursive=True)
numbers = [x[-5] for x in paths]

images = [None]*len(numbers)
for idx, number in enumerate(numbers):
    images[int(number)] = paths[idx]

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
import statistics
'''
for idx in range(8):
    im = Image.fromarray(np.uint8(color[:,:,idx].detach().numpy()*255))
    im.show()
    #ims = Image.fromarray(np.uint8(samples[:, :, idx].detach().numpy() * 255))
    #ims.show()
    print(idx)'''

lr = 1e-4 * 5
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

epochs = 500
errs = []
for epoch in range(epochs):
        pred = model.forward()
        err = mse(pred, samples)
        errs.append(err.item())
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        if epoch % 10 == 0:
            #im = Image.fromarray(np.uint8(pred[:, :, 5].detach().numpy() * 255))
            #im.show()
            print(f'Epoch {epoch} AVG Err {statistics.mean(errs[-10:])}')
        if epoch % 100 == 0:
            print(f'Rough {model.rough.item()} Diffuse {model.diffuse.item()} f0P {model.f0P.item()} Camera {model.camera.detach()}')

for idx in range(8):
    im = Image.fromarray(np.uint8(pred[:,:,idx].detach().numpy()*255))
    im.show()

print('TheEnd')