import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib
from models import OptimizeParameters
from utils import *
from optimize_parameters import optimizeParameters
from matplotlib.widgets import TextBox
from torch.nn.parameter import Parameter

rough, diffuse, relectance = 0.2, 0.2, 0.2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

camera, lights, surface = getSceneLocations(batch=1)

samples = getRealSamples('realSamples1')

gfm = getGfm()
L= 0

model = OptimizeParameters(surface, (lights,False), camera,
                               shadowing=False,
                               rough=rough, diffuse=diffuse, reflectance=relectance)
model.eval()

pred = model.forward()

matplotlib.use('TkAgg')

plt.rcParams["figure.figsize"] = [7.50, 4.50]
plt.rcParams["figure.autolayout"] = False
fig, (ax1, ax2) = plt.subplots(1,2)
plt.subplots_adjust(bottom=0.4)

height_profile_x_gfm, height_profile_y_gfm = getHeightProfile(gfm[0,...,L,0])
height_profile_x_true, height_profile_y_true = getHeightProfile(samples[0,...,L])
height_profile_x_pred, height_profile_y_pred = getHeightProfile(pred[0,...,L])

x = np.linspace(0, len(height_profile_x_gfm) - 1, len(height_profile_x_gfm))
y = np.linspace(0, len(height_profile_y_gfm) - 1, len(height_profile_y_gfm))

ax1.plot(x, height_profile_x_gfm, label='gfm', color='red')
ax1.plot(x, height_profile_x_true, label='ground truth', color='green')
ax1.plot(x, height_profile_x_pred, label='prediction', color='blue')
ax1.set(xlabel='pixels', ylabel='height')
ax1.legend()
ax1.set_title('profile in x-direction')

ax2.plot(y, height_profile_y_gfm, label='gfm', color='red')
ax2.plot(y, height_profile_y_true, label='ground truth', color='green')
ax2.plot(y, height_profile_y_pred, label='prediction', color='blue')
ax2.set(xlabel='pixels', ylabel='height')
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

axstart = plt.axes([0.55, 0.01, 0.3, 0.075])
start = Button(axstart, '->Start Optimization<-', color='yellow')

syn = plt.axes([0.6, 0.16, 0.25, 0.05])
SynB = Button(syn, 'Synthetic', color='red')
SynB.__setattr__('value', False)
SynB.hovercolor = 'green'

para_lights = plt.axes([0.6, 0.22, 0.25, 0.05])
LP = Button(para_lights, 'lights are paramerters', color='green')
LP.__setattr__('value', True)
LP.hovercolor = 'red'


itbox = fig.add_axes([0.2, 0.01, 0.3, 0.075])
iterations = TextBox(itbox, "Iterations", textalignment="center")
iterations.set_val("50000")

path_results = os.path.join('results', 'optimization', '0')
res_path_box = fig.add_axes([0.2, 0.09, 0.65, 0.05])
PathResults = TextBox(res_path_box, "Folder results", textalignment="center")
PathResults.set_val(path_results)

def update(val):

    model.rough = Parameter(torch.tensor(Rslider.val))
    model.reflectance = Parameter(torch.tensor(Fslider.val))
    model.diffuse = Parameter(torch.tensor(Dslider.val))
    pred = model.forward()

    height_profile_x_pred, height_profile_y_pred = getHeightProfile(pred[0, ..., int(radio.value_selected)])

    x = np.linspace(0, len(height_profile_x_gfm) - 1, len(height_profile_x_gfm))
    y = np.linspace(0, len(height_profile_y_gfm) - 1, len(height_profile_y_gfm))

    ax1.lines[2].remove()

    ax1.plot(x, height_profile_x_pred, label='prediction', color='blue')
    ax1.set(xlabel='pixels', ylabel='height')
    ax1.legend()
    ax1.set_title('profile in x-direction')

    ax2.lines[2].remove()

    ax2.plot(y, height_profile_y_pred, label='prediction', color='blue')
    ax2.set(xlabel='pixels', ylabel='height')
    ax2.legend()
    ax2.set_title('profile in y-direction')

    fig.canvas.draw_idle()

def update_L(val):
    L = int(val)
    pred = model.forward()

    height_profile_x_gfm, height_profile_y_gfm = getHeightProfile(gfm[0, ..., L, 0])
    height_profile_x_true, height_profile_y_true = getHeightProfile(samples[0, ..., L])
    height_profile_x_pred, height_profile_y_pred = getHeightProfile(pred[0, ..., L])

    x = np.linspace(0, len(height_profile_x_gfm) - 1, len(height_profile_x_gfm))
    y = np.linspace(0, len(height_profile_y_gfm) - 1, len(height_profile_y_gfm))

    ax1.lines[2].remove()
    ax1.lines[1].remove()
    ax1.lines[0].remove()

    ax1.plot(x, height_profile_x_gfm, label='gfm', color='red')
    ax1.plot(x, height_profile_x_true, label='ground truth', color='green')
    ax1.plot(x, height_profile_x_pred, label='prediction', color='blue')
    ax1.set(xlabel='pixels', ylabel='height')
    ax1.legend()
    ax1.set_title('profile in x-direction')

    ax2.lines[2].remove()
    ax2.lines[1].remove()
    ax2.lines[0].remove()

    ax2.plot(y, height_profile_y_gfm, label='gfm', color='red')
    ax2.plot(y, height_profile_y_true, label='ground truth', color='green')
    ax2.plot(y, height_profile_y_pred, label='prediction', color='blue')
    ax2.set(xlabel='pixels', ylabel='height')
    ax2.legend()
    ax2.set_title('profile in y-direction')

    fig.canvas.draw_idle()

def start_optimization(val):
    print('1) optimize scene parameters for training')

    rough = Rslider.val
    diffuse = Dslider.val
    reflactance = Fslider.val

    its = int(iterations.text) + 1
    path_results = PathResults.text

    start.color = 'purple'
    start.label = 'optimization is running...'

    selected_lights = SelectedLights.value_selected

    plt.close()

    parameters = optimizeParameters(path_real_samples='realSamples1', path_results=path_results, para_lights=LP.value,
                                    iterations=its,
                                    rough=rough, diffuse=diffuse, reflectance=reflactance, selected_lights=selected_lights)


def change_synthetic(val):
    if SynB.value:
        SynB.value=False
        SynB.color='red'
        SynB.hovercolor='green'
    else:
        SynB.value=True
        SynB.color = 'green'
        SynB.hovercolor = 'red'

def change_light_parameter(val):
    if LP.value:
        LP.value=False
        LP.color='red'
        LP.hovercolor='green'
    else:
        LP.value=True
        LP.color = 'green'
        LP.hovercolor = 'red'

Rslider.on_changed(update)
Fslider.on_changed(update)
Dslider.on_changed(update)
radio.on_clicked(update_L)
start.on_clicked(start_optimization)
SynB.on_clicked(change_synthetic)
LP.on_clicked(change_light_parameter)

plt.show()