import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from models import OptimizeParameters
import cv2
import torch
import os
from utils import *
from optimize_parameters import optimizeParameters

rough, diffuse, relectance = 0.334, 0.597, 0.793
intensity = 53.2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

camera, lights, mesh = get_scene_locations(batch_real_samples=1)

samples = get_real_samples('realSamples1')

mean_intensity = get_light_attenuation()
#light_attenuation = torch.ones_like(light_attenuation)
L= 0



model = OptimizeParameters((mesh,True), (lights,False), (camera,False), shadowing=False,
                               device=device, mean_intensity=mean_intensity,
                               rough=rough, diffuse=diffuse, reflectance=relectance,
                               par_r=False, par_d=False, par_ref=False,
                               get_para=False, intensity=intensity)
model.eval()

pred = model.forward()

matplotlib.use('TkAgg')

plt.rcParams["figure.figsize"] = [7.50, 4.50]
plt.rcParams["figure.autolayout"] = False
fig, (ax1, ax2) = plt.subplots(1,2)
plt.subplots_adjust(bottom=0.4)

height_profile_x_la, height_profile_y_la = get_height_profile(mean_intensity[0,...,L,0])
height_profile_x_true, height_profile_y_true = get_height_profile(samples[0,...,L])
height_profile_x_pred, height_profile_y_pred = get_height_profile(pred[0,...,L])

x = np.linspace(0, len(height_profile_x_la) - 1, len(height_profile_x_la))
y = np.linspace(0, len(height_profile_y_la) - 1, len(height_profile_y_la))

ax1.plot(x, height_profile_x_la, label='mean_intensity', color='red')
ax1.plot(x, height_profile_x_true, label='true', color='green')
ax1.plot(x, height_profile_x_pred, label='pred', color='blue')
ax1.set(xlabel='pixels', ylabel='height')
ax1.legend()
ax1.set_title('profile in x-direction')

ax2.plot(y, height_profile_y_la, label='mean_intensity', color='red')
ax2.plot(y, height_profile_y_true, label='true', color='green')
ax2.plot(y, height_profile_y_pred, label='pred', color='blue')
ax2.set(xlabel='pixels', ylabel='height')
ax2.legend()
ax2.set_title('profile in y-direction')

axcolor = 'yellow'
rough_slider = plt.axes([0.20, 0.11, 0.65, 0.03], facecolor=axcolor)
reflectance_slider = plt.axes([0.20, 0.15, 0.65, 0.03], facecolor=axcolor)
diffuse_slider = plt.axes([0.20, 0.19, 0.65, 0.03], facecolor=axcolor)
intensity_slider = plt.axes([0.20, 0.23, 0.65, 0.03], facecolor=axcolor)
Rslider = Slider(rough_slider, 'Rough', 0.0, 1.0, valinit=rough)
Fslider = Slider(reflectance_slider, 'reflectance', 0.0, 1.0, valinit=relectance)
Dslider = Slider(diffuse_slider, 'Diffuse', 0.0, 1.0, valinit=diffuse)
Islider = Slider(intensity_slider, 'Intensity', 0.0, 100.0, valinit=intensity)

rax = plt.axes([0.01, 0.4, 0.06, 0.4], facecolor=axcolor) #[left, bottom, width, height]
radio = RadioButtons(rax, ('0','1','2','3','4','5','6','7','8','9','10','11'))

axstart = plt.axes([0.6, 0.01, 0.3, 0.075])
start = Button(axstart, 'Start Optimization')

def update(val):
    rough = Rslider.val
    reflectance = Fslider.val
    diffuse = Dslider.val
    intensity = Islider.val
    model.rough = torch.tensor(rough)
    model.reflectance = torch.tensor(reflectance)
    model.diffuse = torch.tensor(diffuse)
    model.intensity = torch.tensor(intensity)
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

    height_profile_x_la, height_profile_y_la = get_height_profile(mean_intensity[0, ..., L, 0])
    height_profile_x_true, height_profile_y_true = get_height_profile(samples[0, ..., L])
    height_profile_x_pred, height_profile_y_pred = get_height_profile(pred[0, ..., L])

    x = np.linspace(0, len(height_profile_x_la) - 1, len(height_profile_x_la))
    y = np.linspace(0, len(height_profile_y_la) - 1, len(height_profile_y_la))

    ax1.lines[2].remove()
    ax1.lines[1].remove()
    ax1.lines[0].remove()

    ax1.plot(x, height_profile_x_la, label='mean_intensity', color='red')
    ax1.plot(x, height_profile_x_true, label='true', color='green')
    ax1.plot(x, height_profile_x_pred, label='pred', color='blue')
    ax1.set(xlabel='pixels', ylabel='height')
    ax1.legend()
    ax1.set_title('profile in x-direction')

    ax2.lines[2].remove()
    ax2.lines[1].remove()
    ax2.lines[0].remove()

    ax2.plot(y, height_profile_y_la, label='mean_intensity', color='red')
    ax2.plot(y, height_profile_y_true, label='true', color='green')
    ax2.plot(y, height_profile_y_pred, label='pred', color='blue')
    ax2.set(xlabel='pixels', ylabel='height')
    ax2.legend()
    ax2.set_title('profile in y-direction')

    fig.canvas.draw_idle()

def start_optimization(val):
    print('1) optimize scene parameters for training')

    rough = (0.2, Rslider.val, True)  # , (0.5,0.4,True), (0.8,0.4,True)]
    diffuse = (0.2, Dslider.val, True)  # , (0.5,0.4,True), (0.8,0.4,True)]
    reflactance = (0.2, Fslider.val, True)  # , (0.5,0.4,True), (0.8,0.4,True)]
    intensity = Islider.val

    plt.close()

    parameters = optimizeParameters(path_target='realSamples1', path_results=os.path.join('results', 'optimization',
                                                                                          'rough-diffuse-reflactance-surface-Test'),
                                    epochs=20001, intensity=intensity, weight_decay=0.0, mean_intensity=mean_intensity,
                                    # (synthetic, initial, parameter)
                                    rough=rough, diffuse=diffuse, reflectance=reflactance,
                                    synthetic=False, surface_opimization=True, quick_search=False, plot_every=2000)


Rslider.on_changed(update)
Fslider.on_changed(update)
Dslider.on_changed(update)
Islider.on_changed(update)
radio.on_clicked(update_L)
start.on_clicked(start_optimization)

plt.show()