import glob
import os
import imageio
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np

def get_light_attenuation():
    convert_tensor = transforms.ToTensor()

    file_path = os.path.join('Sensor_2', '**', f'*.jpg')
    paths = glob.glob(file_path, recursive=True)

    images={}
    for idx, path in enumerate(paths):
        number = path[-6:-4]
        if number[0] == '/':
            number = number[1]
        number = int(number)
        im = imageio.imread(path)
        im = convert_tensor(Image.fromarray(im))[0].unsqueeze(-1)
        if number not in images:
            images[number] = im
        else:
            images[number] = torch.cat((images[number], im), dim=-1)



    light_attenuation = None
    for im_nr in sorted(images.keys()) :
        im = torch.from_numpy(gaussian_filter(torch.median(images[im_nr], dim=2)[0], sigma=5, mode='reflect'))
        images[im_nr] = im / im.max()
        if light_attenuation == None:
            light_attenuation = images[im_nr].unsqueeze(-1)
        else:
            light_attenuation = torch.cat((light_attenuation, images[im_nr].unsqueeze(-1)), dim=-1)

    return light_attenuation.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # S,C,H,W,L,1

# light_attenuation = get_light_attenuation()



'''for im_nr in range(12):
    plt.imshow(light_attenuation[:,:,im_nr].cpu().detach().numpy())
    plt.show()
    print(f'{im_nr}: Max {light_attenuation[:,:,im_nr].max()} Min {light_attenuation[:,:,im_nr].min()}')
    bla = gaussian_filter(light_attenuation[:,:,im_nr], sigma=10, mode='reflect')
    plt.imshow(bla)
    print(f'{im_nr}.2: Max {bla.max()} Min {bla.min()}')

    plt.show()

print('TheEnd')'''