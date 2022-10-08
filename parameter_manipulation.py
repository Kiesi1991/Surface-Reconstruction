import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from models import OptimizeParameters
from optimize_parameters import optimizeParameters
from matplotlib.widgets import TextBox
from torch.nn.parameter import Parameter
from shaders import *
from utils import *

rough, diffuse, relectance = 0.8, 0.8, 0.8
path_results = os.path.join('results', 'optimization', '4')
its = "50000"
samples = getRealSamples('realSamples1')
L= 0

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

camera, lights, surface = getScene(batch=1)

model = OptimizeParameters(surface.to(device), lights.to(device), camera.to(device),
                               shadowing=False,
                               rough=rough, diffuse=diffuse, reflectance=relectance)
model.eval()

syn_samples = model.create_synthetic_images()

pred = model.forward()

matplotlib.use('TkAgg')

plt.rcParams["figure.figsize"] = [7.50, 4.50]
plt.rcParams["figure.autolayout"] = False
fig, (ax1, ax2) = plt.subplots(1,2)
plt.subplots_adjust(bottom=0.4)

height_profile_x_gfm, height_profile_y_gfm = getHeightProfile(model.gfm[0,...,L,0])
height_profile_x_true, height_profile_y_true = getHeightProfile(samples[0,...,L])
height_profile_x_pred, height_profile_y_pred = getHeightProfile(pred[0,...,L])

x = np.linspace(0, len(height_profile_x_gfm) - 1, len(height_profile_x_gfm))
y = np.linspace(0, len(height_profile_y_gfm) - 1, len(height_profile_y_gfm))

ax1.plot(x, height_profile_x_gfm, label='gfm', color='red')
ax1.plot(x, height_profile_x_true, label='ground truth', color='green')
ax1.plot(x, height_profile_x_pred, label='prediction', color='blue')
ax1.set(xlabel='pixels')
ax1.legend()
ax1.set_title('profile in x-direction')

ax2.plot(y, height_profile_y_gfm, label='gfm', color='red')
ax2.plot(y, height_profile_y_true, label='ground truth', color='green')
ax2.plot(y, height_profile_y_pred, label='prediction', color='blue')
ax2.set(xlabel='pixels')
ax2.legend()
ax2.set_title('profile in y-direction')

axcolor = 'yellow'
rough_slider = plt.axes([0.20, 0.16, 0.3, 0.03], facecolor=axcolor)
reflectance_slider = plt.axes([0.20, 0.2, 0.3, 0.03], facecolor=axcolor)
diffuse_slider = plt.axes([0.20, 0.24, 0.3, 0.03], facecolor=axcolor)
Rslider = Slider(rough_slider, 'Rough', 0.0, 1.0, valinit=rough)
Fslider = Slider(reflectance_slider, 'reflectance', 0.0, 1.0, valinit=relectance)
Dslider = Slider(diffuse_slider, 'Diffuse', 0.0, 1.0, valinit=diffuse)

rax = plt.axes([0.01, 0.4, 0.06, 0.4], facecolor=axcolor) #[left, bottom, width, height]
radio = RadioButtons(rax, ('0','1','2','3','4','5','6','7','8','9','10','11'))

selected_lights = plt.axes([0.01, 0.01, 0.06, 0.3], facecolor=axcolor) #[left, bottom, width, height]
SelectedLights = RadioButtons(selected_lights, ('all levels', 'level 1', 'level 2', 'level 3', 'level 2+3'))

axstart = plt.axes([0.6, 0.01, 0.3, 0.075])
start = Button(axstart, '->Start Optimization<-', color='yellow')

syn = plt.axes([0.6, 0.16, 0.145, 0.05])
SynB = Button(syn, 'real', color='white')
SynB.__setattr__('value', False)

new_syn = plt.axes([0.9-0.145, 0.16, 0.145, 0.05])
Ns = Button(new_syn, 'new surface', color='blue')
Ns.hovercolor = 'gray'

para_lights = plt.axes([0.9-0.145, 0.22, 0.145, 0.05])
LP = Button(para_lights, 'lights are paramerters', color='green')
LP.__setattr__('value', True)
LP.hovercolor = 'red'

image_view = plt.axes([0.8, 0.30, 0.1, 0.04]) #[left, bottom, width, height]
IV = Button(image_view, 'image view', color='red')
IV.__setattr__('value', False)
IV.hovercolor = 'green'

reg = plt.axes([0.375, 0.07, 0.17, 0.05]) #[left, bottom, width, height]
REG = Button(reg, 'abs', color='white')
REG.__setattr__('value', 'abs')

itbox = fig.add_axes([0.6, 0.22, 0.145, 0.05])
iterations = TextBox(itbox, "Iterations", textalignment="center")
iterations.set_val(its)

res_path_box = fig.add_axes([0.6, 0.11, 0.3, 0.04])
PathResults = TextBox(res_path_box, "Results", textalignment="center")
PathResults.set_val(path_results)

# hyperparameter for surface creation
pbox = fig.add_axes([0.2, 0.01, 0.05, 0.05]) #[left, bottom, width, height]
pPara = TextBox(pbox, "p ", textalignment="center")
pPara.set_val(0.008)

Hbox = fig.add_axes([0.2+0.07, 0.01, 0.05, 0.05]) #[left, bottom, width, height]
HPara = TextBox(Hbox, "H ", textalignment="center")
HPara.set_val(0.01)

Ibox = fig.add_axes([0.2+0.14, 0.01, 0.05, 0.05]) #[left, bottom, width, height]
IPara = TextBox(Ibox, "IT ", textalignment="center")
IPara.set_val(2)

lbox = fig.add_axes([0.2+0.21, 0.01, 0.05, 0.05]) #[left, bottom, width, height]
lPara = TextBox(lbox, "l ", textalignment="center")
lPara.set_val(100)

hbox = fig.add_axes([0.2+0.28, 0.01, 0.05, 0.05]) #[left, bottom, width, height]
hPara = TextBox(hbox, "h ", textalignment="center")
hPara.set_val(150)

sigmasbox = fig.add_axes([0.2, 0.07, 0.17, 0.05]) #[left, bottom, width, height]
sigmasPara = TextBox(sigmasbox, "sigmas ", textalignment="center")
sigmasPara.set_val('10,6,3,1.5,1')


def update(val):

    model.rough = Parameter(torch.tensor(Rslider.val))
    model.reflectance = Parameter(torch.tensor(Fslider.val))
    model.diffuse = Parameter(torch.tensor(Dslider.val))
    pred = model.forward()
    syn_samples = model.create_synthetic_images()

    if SynB.value:
        height_profile_x_true, height_profile_y_true = getHeightProfile(syn_samples[0, ..., int(radio.value_selected)])
    else:
        height_profile_x_true, height_profile_y_true = getHeightProfile(samples[0, ..., int(radio.value_selected)])
    height_profile_x_pred, height_profile_y_pred = getHeightProfile(pred[0, ..., int(radio.value_selected)])

    height_profile_x_gfm, height_profile_y_gfm = getHeightProfile(model.gfm[0, ..., int(radio.value_selected), 0])

    x = np.linspace(0, len(height_profile_x_gfm) - 1, len(height_profile_x_gfm))
    y = np.linspace(0, len(height_profile_y_gfm) - 1, len(height_profile_y_gfm))

    ax1.lines[2].remove()
    ax1.lines[1].remove()
    ax1.lines[0].remove()

    ax1.plot(x, height_profile_x_gfm, label='gfm', color='red')
    ax1.plot(x, height_profile_x_true, label='ground truth', color='green')
    ax1.plot(x, height_profile_x_pred, label='prediction', color='blue')
    ax1.set(xlabel='pixels')
    ax1.legend()
    ax1.set_title('profile in x-direction')

    try:
        ax2.lines[2].remove()
        ax2.lines[1].remove()
        ax2.lines[0].remove()
    except: pass

    try: ax2.images[0].remove()
    except: pass

    if IV.value:
        ax2.set_title('image sample')
        ax2.imshow(syn_samples[0, 0, ..., int(radio.value_selected)].cpu().detach().numpy() if SynB.value
                   else samples[0, 0, ..., int(radio.value_selected)].cpu().detach().numpy())
        try: ax2.get_legend().remove()
        except: pass
    else:
        ax2.plot(y, height_profile_y_gfm, label='gfm', color='red')
        ax2.plot(y, height_profile_y_true, label='ground truth', color='green')
        ax2.plot(y, height_profile_y_pred, label='prediction', color='blue')
        ax2.set(xlabel='pixels')
        ax2.legend()
        ax2.set_title('profile in y-direction')

    # recompute the ax.dataLim
    ax1.relim()
    ax2.relim()
    # update ax.viewLim using the new dataLim
    ax1.autoscale_view()
    ax2.autoscale_view()

    fig.canvas.draw_idle()

def start_optimization(val):
    print('1) optimize scene parameters for training')

    its = int(iterations.text) + 1
    path_results = PathResults.text

    start.color = 'purple'
    start.label = 'optimization is running...'

    selected_lights = SelectedLights.value_selected

    plt.close()

    model.lights = Parameter(model.lights) if LP.value else model.lights
    model.shadowing = True

    parameters = optimizeParameters(model, path_results=path_results, regularization_function=REG.value,
                                    iterations=its, selected_lights=selected_lights)

def change_synthetic(val):
    if SynB.value:
        SynB.value=False
        SynB.label.set_text("real")
        model.synthetic = False
        update(val)
    else:
        SynB.value=True
        SynB.label.set_text("synthetic")
        model.synthetic = True
        update(val)

def change_regularization(val):
    if REG.value == 'abs':
        REG.value = 'square'
        REG.label.set_text("square")
    elif REG.value == 'square':
        REG.value = 'exp'
        REG.label.set_text("exp")
    elif REG.value == 'exp':
        REG.value = 'abs'
        REG.label.set_text("abs")
    else:
        raise('regularization is not defined!')

def change_light_parameter(val):
    if LP.value:
        LP.value=False
        LP.color='red'
        model.lights.requires_grad = False
        LP.hovercolor='green'
    else:
        LP.value=True
        LP.color = 'green'
        model.lights.requires_grad = True
        LP.hovercolor = 'red'

def new_syn_surface(val):
    model.synthetic_surface = createSurface(resolution=(surface.shape[1], surface.shape[2]),
                                            sigmas=[float(e) for e in (sigmasPara.text.split(','))], p=float(pPara.text), H=float(HPara.text),
                                            I=int(IPara.text), l=int(lPara.text), h=int(hPara.text)).to(device).unsqueeze(0)
    update(val)

def change_view(val):
    if not IV.value:
        IV.value = True
        IV.color = 'green'
    else:
        print('set image view off is not possible!')
    update(val)

Rslider.on_changed(update)
Fslider.on_changed(update)
Dslider.on_changed(update)
radio.on_clicked(update)
start.on_clicked(start_optimization)
synthetic_surface = SynB.on_clicked(change_synthetic)
Ns.on_clicked(new_syn_surface)
IV.on_clicked(change_view)
LP.on_clicked(change_light_parameter)
REG.on_clicked(change_regularization)

plt.show()