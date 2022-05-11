import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from models import OptimizeParameters
import cv2
import torch
import os
from utils import get_scene_parameters

path = os.path.join('results', 'optimization', '0', 'Epoch-3000')
parameters = get_scene_parameters(path)

model = OptimizeParameters((parameters['surface'][0].unsqueeze(0), False), (parameters['lights'], False), (parameters['camera'], False), par_li=False,
                 par_r=False, par_d=False, par_f0=False,
                 par_x=False, par_y=False, get_para=False)
model.eval()
im = model.forward()

im2 = cv2.cvtColor(im[:,:,0].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

matplotlib.use('TkAgg')

plt.rcParams["figure.figsize"] = [7.50, 4.50]
plt.rcParams["figure.autolayout"] = False
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
img = ax.imshow(im2)
axcolor = 'yellow'
rough_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
f0_slider = plt.axes([0.20, 0.05, 0.65, 0.03], facecolor=axcolor)
diffuse_slider = plt.axes([0.20, 0.09, 0.65, 0.03], facecolor=axcolor)
intensity_slider = plt.axes([0.20, 0.13, 0.65, 0.03], facecolor=axcolor)
Rslider = Slider(rough_slider, 'Rough', 0.0, 1.0, valinit=0.5)
Fslider = Slider(f0_slider, 'f0', 0.0, 1.0, valinit=0.5)
Dslider = Slider(diffuse_slider, 'Diffuse', 0.0, 1.0, valinit=0.5)
Islider = Slider(intensity_slider, 'Intensity', 0.0, 10.0, valinit=3.0)

rax = plt.axes([0.05, 0.4, 0.1, 0.4], facecolor=axcolor) #[left, bottom, width, height]
radio = RadioButtons(rax, ('0','1','2','3','4','5','6','7','8','9','10','11'))

def update(val):
    rough = Rslider.val
    f0 = Fslider.val
    diffuse = Dslider.val
    intensity = Islider.val
    model.rough = torch.nn.Parameter(torch.tensor(rough))
    model.f0P = torch.nn.Parameter(torch.tensor(f0))
    model.diffuse = torch.nn.Parameter(torch.tensor(diffuse))
    model.intensity = torch.nn.Parameter(torch.tensor(intensity))
    im = model.forward()
    im2 = cv2.cvtColor(im[:, :, int(radio.value_selected)].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
    ax.imshow(im2)
    fig.canvas.draw_idle()

def update_L(val):
    L = int(val)
    im = model.forward()
    im2 = cv2.cvtColor(im[:, :, L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
    ax.imshow(im2)
    fig.canvas.draw_idle()

Rslider.on_changed(update)
Fslider.on_changed(update)
Dslider.on_changed(update)
Islider.on_changed(update)
radio.on_clicked(update_L)

plt.show()