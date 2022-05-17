import torch
import glob, os
import imageio
from PIL import Image
from torchvision import transforms
from models import OptimizeParameters
import matplotlib.pyplot as plt
import cv2
import numpy as np
import statistics
from utils import *
from tqdm import tqdm
from shaders import FilamentShading

def optimizeParameters(path_target='realSamples', path_results=os.path.join('results', 'optimization'),
                       lr=1e-3, epochs=3001,
                       intensity=2.5,
                       rough=(0.1,0.1), diffuse=(0.1,0.1), f0P=(0.9,0.9),
                       par_r=True, par_d=True, par_f0=True,
                       synthetic=False):

    samples = get_real_samples(path_target)

    camera, lights, mesh = get_scene_locations(batch_real_samples=1) if synthetic else get_scene_locations(batch_real_samples=samples.shape[0])
    path = create_next_folder(path_results)

    if synthetic:
        light_intensity = torch.ones((1,1,1,1,12,1))
        shader = FilamentShading(camera, lights, light_intensity, intensity=torch.tensor(intensity),
                                 rough=torch.tensor(rough[0]), diffuse=torch.tensor(diffuse[0]), f0P=torch.tensor(f0P[0]))
        surface = torch.tensor(createSurface((386, 516)).tolist()).unsqueeze(0)
        samples = shader.forward(surface).permute((0,2,3,1)).unsqueeze(0)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = OptimizeParameters((mesh,True), (lights,False), (camera,False), device=device,
                               par_r=par_r, par_d=par_d, par_f0=par_f0,
                               get_para=False, intensity=intensity)

    parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(name)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    errs = []
    errors = []
    roughs = []
    diffuses = []
    f0s = []
    intensities = []
    #TimeEpoch = TimeDelta()
    #TimePlots = TimeDelta()
    for epoch in tqdm(range(epochs)):
            pred = model.forward()
            err = mse(pred.to(device), samples.to(device)) #+ 0.001 * (model.mesh**2).sum()
            errs.append(err.item())
            optimizer.zero_grad()
            err.backward()
            #torch.nn.utils.clip_grad_value_(model.mesh, 0.001)
            optimizer.step()
            '''if epoch==epochs//2:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr/10, weight_decay=0.0)
                print(f'New Lr {lr/10}')'''
            if epoch % 50 == 0:
                # print(f'Epoch {epoch} AVG Err {statistics.mean(errs[-10:])} Surface Max {model.mesh.detach().max()} Surface Min {model.mesh.detach().min()}')
                errors.append(statistics.mean(errs[-10:]))
                #epoch_time = TimeEpoch.event()
                #print(f'Epoch {epoch} / {epochs} - Time per Epoch {epoch_time/50}')
            if epoch % 4000 == 0:
                #_ = TimePlots.event()
                os.mkdir(os.path.join(path, f'Epoch-{epoch}'))
                path2 = os.path.join(path, f'Epoch-{epoch}')
                #print(f'Rough {model.rough.item()} Diffuse {torch.sigmoid(model.diffuse).item()} f0P {model.f0P.item()}')
                num_L = samples.shape[4]
                for L in range(num_L):
                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)

                    plt.imshow(samples[0,0,...,L].cpu().detach().numpy())
                    plt.clim(0, 1.0)

                    plt.subplot(1, 2, 2)

                    plt.imshow(pred[0,0,...,L].cpu().detach().numpy())
                    plt.clim(0, 1.0)

                    plt.savefig(os.path.join(path2, f'{L}.png'))
                    plt.close()

                    p = cv2.cvtColor(pred[0,0,...,L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)
                    t = cv2.cvtColor(samples[0,0,...,L].cpu().detach().numpy(), cv2.COLOR_GRAY2RGB)

                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)
                    plt.imshow(t)
                    plt.clim(0, 1.0)

                    plt.subplot(1, 2, 2)
                    plt.imshow(p)
                    plt.clim(0, 1.0)

                    plt.savefig(os.path.join(path2, f'TrueRGB-{L}.png'))
                    plt.close()

                x = np.linspace(0, len(errors) - 1, len(errors))
                plt.plot(x, errors, label='errors')
                plt.xlabel('epoch')
                plt.ylabel('Error')
                plt.legend()
                plt.savefig(os.path.join(path, f'error.png'))
                plt.close()

                roughs.append(model.rough.cpu().detach().numpy().item())
                diffuses.append(model.diffuse.cpu().detach().numpy().item())
                f0s.append(model.f0P.cpu().detach().numpy().item())
                intensities.append(model.intensity.cpu().detach().numpy().item())

                x = np.linspace(0, len(roughs) - 1, len(roughs))
                plt.plot(x, roughs, label='rough')
                plt.plot(x, diffuses, label='diffuse')
                plt.plot(x, f0s, label='f0')
                #plt.plot(x, intensities, label='intensity')
                plt.xlabel('epoch')
                plt.ylabel('value')
                plt.title(f'parameters with constant intensity {model.intensity.cpu().detach().numpy().item()}')
                plt.legend()
                plt.savefig(os.path.join(path, f'parameters.png'))
                plt.close()

                if synthetic:
                    surface_line = surface.cpu().detach().numpy()[0, 200, :]
                    pred_surface_line = model.mesh.cpu().detach().numpy()[0, 200, :]

                    x = np.linspace(0, len(surface_line) - 1, len(surface_line))
                    plt.plot(x, surface_line, label='ground truth')
                    plt.plot(x, pred_surface_line, label='prediction')
                    plt.xlabel('x')
                    plt.ylabel('height')
                    plt.title(f'Synthetic: (rough,diffuse,f0)={(rough[0],diffuse[0],f0P[0])}; \n'
                              f'Initial value vor prediction: (rough,diffuse,f0)={(rough[1],diffuse[1],f0P[1])}')
                    plt.legend()
                    plt.savefig(os.path.join(path, f'compare-height-{epoch}.png'))
                    plt.close()

                with open(os.path.join(path2, 'parameters.txt'), 'w') as f:
                    f.write(f'Parameters {parameters}\n'
                            f'Rough {model.rough.item()} Diffuse {model.diffuse.item()} f0P {model.f0P.item()} \n'
                            f'Camera {model.camera.detach()}\n'
                            f'Lights {model.lights.detach()}\n'
                            f'Surface Max {model.mesh.detach().max()}'
                            f'Surface min {model.mesh.detach().min()}\n'
                            f'Light Intensity {model.light_intensity.detach()}\n'
                            f'Intensity {model.intensity.detach()}\n'
                            f'X {model.x.detach()}\n'
                            f'Y {model.y.detach()}\n'
                            f'AVG Err {statistics.mean(errs[-10:])}')

                torch.save(model.rough.detach().cpu(), os.path.join(path2, 'rough.pt'))
                torch.save(model.diffuse.detach().cpu(), os.path.join(path2, 'diffuse.pt'))
                torch.save(model.f0P.detach().cpu(), os.path.join(path2, 'f0P.pt'))
                torch.save(model.camera.detach().cpu(), os.path.join(path2, 'camera.pt'))
                torch.save(model.lights.detach().cpu(), os.path.join(path2, 'lights.pt'))
                torch.save(model.light_intensity.detach(), os.path.join(path2, 'light_intensity.pt'))
                torch.save(model.intensity.detach(), os.path.join(path2, 'intensity.pt'))
                torch.save(model.mesh.detach(), os.path.join(path2, 'surface.pt'))
                torch.save(model.x.detach(), os.path.join(path2, 'x.pt'))
                torch.save(model.y.detach(), os.path.join(path2, 'y.pt'))

                #plotting_time = TimePlots.event()
                #print(f'Plotting Time {plotting_time}')

    return {'rough' : model.rough.detach().cpu(),
            'diffuse' : model.diffuse.detach().cpu(),
            'f0P' : model.f0P.detach().cpu(),
            'camera' : model.camera.detach().cpu(),
            'lights' : model.lights.detach().cpu(),
            'light_intensity': model.light_intensity.detach().cpu(),
            'intensity': model.intensity.detach().cpu(),
            'x': model.x.detach().cpu(),
            'y': model.x.detach().cpu()}