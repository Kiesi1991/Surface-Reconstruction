import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from models import OptimizeParameters
import cv2
import torch
import os
from utils import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

camera, lights, mesh = get_scene_locations(batch_real_samples=1)

samples = get_real_samples('realSamples1')

light_attenuation = get_light_attenuation()
L= 0



model = OptimizeParameters((mesh,True), (lights,False), (camera,False),
                               device=device, light_attenuation=light_attenuation,
                               rough=0.4, diffuse=0.4, reflectance=0.4,
                               par_r=False, par_d=False, par_ref=False,
                               get_para=False, intensity=5.0)
model.eval()

pred = model.forward()

matplotlib.use('TkAgg')

plt.rcParams["figure.figsize"] = [7.50, 4.50]
plt.rcParams["figure.autolayout"] = False
fig, (ax1, ax2) = plt.subplots(1,2)
plt.subplots_adjust(bottom=0.25)

height_profile_x_la, height_profile_y_la = get_height_profile(light_attenuation[0,...,L,0])
height_profile_x_true, height_profile_y_true = get_height_profile(samples[0,...,L])
height_profile_x_pred, height_profile_y_pred = get_height_profile(pred[0,...,L])

x = np.linspace(0, len(height_profile_x_la) - 1, len(height_profile_x_la))
y = np.linspace(0, len(height_profile_y_la) - 1, len(height_profile_y_la))

ax1.plot(x, height_profile_x_la, label='la', color='red')
ax1.plot(x, height_profile_x_true, label='true', color='green')
ax1.plot(x, height_profile_x_pred, label='pred', color='blue')
ax1.set(xlabel='pixels', ylabel='height')
ax1.legend()
ax1.set_title('profile in x-direction')

ax2.plot(y, height_profile_y_la, label='la', color='red')
ax2.plot(y, height_profile_y_true, label='true', color='green')
ax2.plot(y, height_profile_y_pred, label='pred', color='blue')
ax2.set(xlabel='pixels', ylabel='height')
ax2.legend()
ax2.set_title('profile in y-direction')

axcolor = 'yellow'
rough_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
reflectance_slider = plt.axes([0.20, 0.05, 0.65, 0.03], facecolor=axcolor)
diffuse_slider = plt.axes([0.20, 0.09, 0.65, 0.03], facecolor=axcolor)
intensity_slider = plt.axes([0.20, 0.13, 0.65, 0.03], facecolor=axcolor)
Rslider = Slider(rough_slider, 'Rough', 0.0, 1.0, valinit=0.5)
Fslider = Slider(reflectance_slider, 'reflectance', 0.0, 1.0, valinit=0.5)
Dslider = Slider(diffuse_slider, 'Diffuse', 0.0, 1.0, valinit=0.5)
Islider = Slider(intensity_slider, 'Intensity', 0.0, 10.0, valinit=3.0)

rax = plt.axes([0.05, 0.4, 0.1, 0.4], facecolor=axcolor) #[left, bottom, width, height]
radio = RadioButtons(rax, ('0','1','2','3','4','5','6','7','8','9','10','11'))

def update(val):
    rough = Rslider.val
    reflectance = Fslider.val
    diffuse = Dslider.val
    intensity = Islider.val
    model.rough = torch.tensor(rough)
    model.reflectance = torch.tensor(reflectance)
    model.diffuse = torch.tensor(diffuse)
    model.intensity = torch.tensor(intensity)#
    pred = model.forward()

    height_profile_x_pred, height_profile_y_pred = get_height_profile(pred[0, ..., int(radio.value_selected)])

    x = np.linspace(0, len(height_profile_x_la) - 1, len(height_profile_x_la))
    y = np.linspace(0, len(height_profile_y_la) - 1, len(height_profile_y_la))

    ax1.lines[2].remove()

    ax1.plot(x, height_profile_x_pred, label='pred', color='blue')
    ax1.set(xlabel='pixels', ylabel='height')
    ax1.legend()
    ax1.set_title('profile in x-direction')

    ax2.lines[2].remove()

    ax2.plot(y, height_profile_y_pred, label='pred', color='blue')
    ax2.set(xlabel='pixels', ylabel='height')
    ax2.legend()
    ax2.set_title('profile in y-direction')

    fig.canvas.draw_idle()

def update_L(val):
    L = int(val)
    pred = model.forward()

    height_profile_x_la, height_profile_y_la = get_height_profile(light_attenuation[0, ..., L, 0])
    height_profile_x_true, height_profile_y_true = get_height_profile(samples[0, ..., L])
    height_profile_x_pred, height_profile_y_pred = get_height_profile(pred[0, ..., L])

    x = np.linspace(0, len(height_profile_x_la) - 1, len(height_profile_x_la))
    y = np.linspace(0, len(height_profile_y_la) - 1, len(height_profile_y_la))

    ax1.lines[2].remove()
    ax1.lines[1].remove()
    ax1.lines[0].remove()

    ax1.plot(x, height_profile_x_la, label='la', color='red')
    ax1.plot(x, height_profile_x_true, label='true', color='green')
    ax1.plot(x, height_profile_x_pred, label='pred', color='blue')
    ax1.set(xlabel='pixels', ylabel='height')
    ax1.legend()
    ax1.set_title('profile in x-direction')

    ax2.lines[2].remove()
    ax2.lines[1].remove()
    ax2.lines[0].remove()

    ax2.plot(y, height_profile_y_la, label='la', color='red')
    ax2.plot(y, height_profile_y_true, label='true', color='green')
    ax2.plot(y, height_profile_y_pred, label='pred', color='blue')
    ax2.set(xlabel='pixels', ylabel='height')
    ax2.legend()
    ax2.set_title('profile in y-direction')

    fig.canvas.draw_idle()

Rslider.on_changed(update)
Fslider.on_changed(update)
Dslider.on_changed(update)
Islider.on_changed(update)
radio.on_clicked(update_L)

plt.show()