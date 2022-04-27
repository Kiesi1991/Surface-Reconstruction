import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from models import OptimizeParameters
import cv2
import torch

model = OptimizeParameters(None,None,None)
model.eval()
im = model.forward()

im = cv2.cvtColor(im[:,:,0].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

matplotlib.use('TkAgg')

plt.rcParams["figure.figsize"] = [7.50, 4.50]
plt.rcParams["figure.autolayout"] = False
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
img = ax.imshow(im)
axcolor = 'yellow'
rough_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
f0_slider = plt.axes([0.20, 0.05, 0.65, 0.03], facecolor=axcolor)
diffuse_slider = plt.axes([0.20, 0.09, 0.65, 0.03], facecolor=axcolor)
Rslider = Slider(rough_slider, 'Rough', 0.0, 1.0, valinit=0.5)
Fslider = Slider(f0_slider, 'f0', 0.0, 1.0, valinit=0.5)
Dslider = Slider(diffuse_slider, 'Diffuse', 0.0, 1.0, valinit=0.5)
def update(val):
    rough = Rslider.val
    f0 = Fslider.val
    diffuse = Dslider.val
    model.rough = torch.nn.Parameter(torch.tensor(rough))
    model.f0P = torch.nn.Parameter(torch.tensor(f0))
    model.diffuse = torch.nn.Parameter(torch.tensor(diffuse))
    im = model.forward()
    im = cv2.cvtColor(im[:, :, 0].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
    ax.imshow(im)
    fig.canvas.draw_idle()
Rslider.on_changed(update)
Fslider.on_changed(update)
Dslider.on_changed(update)
plt.show()