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
                       lr=1e-4, weight_decay=0.,
                       epochs=3001, mean_intensity=1.0,
                       intensity=2.5, selected_lights='all', para_lights=True,
                       rough=(0.5,0.5,True), diffuse=(0.5,0.5,True), reflectance=(0.5,0.5,True),
                       synthetic=False, surface_opimization=True, quick_search=False, plot_every=1000):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(os.path.join(path_results)):
        os.mkdir(os.path.join(path_results))

    samples = get_real_samples(path_target)

    if selected_lights=='bottom':
        start , end = 8, 11
    elif selected_lights=='middle':
        start, end = 4, 7
    elif selected_lights=='top':
        start, end = 0, 3
    elif selected_lights=='middle+top':
        start, end = 0, 7
    else:
        start, end = 0, 11

    camera, lights, mesh = get_scene_locations(batch_real_samples=1) if synthetic else get_scene_locations(batch_real_samples=samples.shape[0])
    path = path_results if quick_search else create_next_folder(path_results)

    if synthetic:
        light_intensity = torch.ones((1,1,1,1,12,1))
        shader = FilamentShading(camera, lights, light_intensity, intensity=torch.tensor(intensity),
                                 rough=torch.tensor(rough[0]), diffuse=torch.tensor(diffuse[0]), reflectance=torch.tensor(reflectance[0]), device=device)
        surface = torch.tensor(createSurface((386, 516)).tolist()).unsqueeze(0).to(device)
        samples = shader.forward(surface).permute((0,2,3,1)).unsqueeze(0)

    model = OptimizeParameters((mesh,True) if surface_opimization else (surface,False),
                               (lights,para_lights), (camera,False), device=device, mean_intensity=mean_intensity,
                               rough=rough[1], diffuse=diffuse[1], reflectance=reflectance[1],
                               par_r=rough[2], par_d=diffuse[2], par_ref=reflectance[2],
                               get_para=False, intensity=intensity)

    '''pred = model.forward()

    for L in range(12):

        height_profile_x_la, height_profile_y_la = get_height_profile(light_attenuation[0,...,L,0])
        height_profile_x_true, height_profile_y_true = get_height_profile(samples[0,...,L])
        height_profile_x_pred, height_profile_y_pred = get_height_profile(pred[0,...,L])

        x = np.linspace(0, len(height_profile_x_la) - 1, len(height_profile_x_la))
        y = np.linspace(0, len(height_profile_y_la) - 1, len(height_profile_y_la))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)

        plt.plot(x, height_profile_x_la, label='la')
        plt.plot(x, height_profile_x_true, label='true')
        plt.plot(x, height_profile_x_pred, label='pred')
        plt.xlabel('pixels')
        plt.ylabel('height')
        plt.legend()
        plt.title('profile in x-direction')

        plt.subplot(1, 2, 2)


        plt.plot(y, height_profile_y_la, label='la')
        plt.plot(y, height_profile_y_true, label='true')
        plt.plot(y, height_profile_y_pred, label='pred')
        plt.xlabel('pixels')
        plt.ylabel('height')
        plt.legend()
        plt.title('profile in y-direction')

        plt.show()
'''
    parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(name)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    errs = []
    errors = []
    roughs = []
    diffuses = []
    reflectances = []
    intensities = []
    #TimeEpoch = TimeDelta()
    #TimePlots = TimeDelta()
    for epoch in tqdm(range(epochs)):
            pred = model.forward()
            err = mse(pred[...,start:end+1].to(device), samples[...,start:end+1].to(device)) #+ 0.001 * (model.rough - model.reflectance) #+ 0.001 * (model.mesh**2).sum()
            errs.append(err.item())
            optimizer.zero_grad()
            err.backward()
            #torch.nn.utils.clip_grad_value_(model.mesh, 0.0001)
            optimizer.step()
            '''if epoch==epochs//2:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr/10, weight_decay=0.0)
                print(f'New Lr {lr/10}')'''
            if epoch % 50 == 0:
                # print(f'Epoch {epoch} AVG Err {statistics.mean(errs[-10:])} Surface Max {model.mesh.detach().max()} Surface Min {model.mesh.detach().min()}')
                errors.append(statistics.mean(errs[-10:]))
                #epoch_time = TimeEpoch.event()
                #print(f'Epoch {epoch} / {epochs} - Time per Epoch {epoch_time/50}')
            if epoch % plot_every == 0:
                #_ = TimePlots.event()

                roughs.append(model.rough.cpu().detach().numpy().item())
                diffuses.append(model.diffuse.cpu().detach().numpy().item())
                reflectances.append(model.reflectance.cpu().detach().numpy().item())
                intensities.append(model.intensity.cpu().detach().numpy().item())

                if not quick_search:
                    os.mkdir(os.path.join(path, f'Epoch-{epoch}'))
                    path2 = os.path.join(path, f'Epoch-{epoch}')
                    #print(f'Rough {model.rough.item()} Diffuse {torch.sigmoid(model.diffuse).item()} reflectance {model.reflectance.item()}')
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

                    if synthetic:
                        height_profile_x_gt, height_profile_y_gt = get_height_profile(surface)
                    height_profile_x_pred, height_profile_y_pred = get_height_profile(model.mesh)

                    x = np.linspace(0, len(height_profile_x_pred) - 1, len(height_profile_x_pred))
                    y = np.linspace(0, len(height_profile_y_pred) - 1, len(height_profile_y_pred))

                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)

                    if synthetic:
                        plt.plot(x, height_profile_x_gt, label='ground truth')
                    plt.plot(x, height_profile_x_pred, label='prediction')
                    plt.xlabel('pixels')
                    plt.ylabel('height')
                    plt.legend()
                    plt.title('profile in x-direction')

                    plt.subplot(1, 2, 2)

                    if synthetic:
                        plt.plot(y, height_profile_y_gt, label='ground truth')
                    plt.plot(y, height_profile_y_pred, label='prediction')
                    plt.xlabel('pixels')
                    plt.ylabel('height')
                    plt.legend()
                    plt.title('profile in y-direction')

                    plt.savefig(os.path.join(path, f'height-profile.png'))
                    plt.close()

                    normal_vectors = getNormals(model.mesh.detach(), x=model.x.detach(), y=model.y.detach())
                    #light_vectors = getVectors(model.mesh.detach(), model.lights.detach().unsqueeze(1).unsqueeze(1).unsqueeze(0), x=model.x.detach(), y=model.y.detach())
                    z_vector = torch.tensor([0.,0.,1.]).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

                    angles = torch.acos((z_vector * normal_vectors).sum(dim=-1, keepdim=True)) * 90 / (torch.pi/2)

                    plt.imshow(angles[0,0,...,0].cpu().detach().numpy())
                    plt.colorbar()
                    plt.savefig(os.path.join(path, f'angles.png'))
                    plt.close()


                    with open(os.path.join(path2, 'parameters.txt'), 'w') as f:
                        f.write(f'Parameters {parameters}\n'
                                f'Rough {model.rough.item()} Diffuse {model.diffuse.item()} Reflectance {model.reflectance.item()} \n'
                                f'Camera {model.camera.detach()}\n'
                                f'Lights {model.lights.detach()}\n'
                                f'Surface Max {model.mesh.detach().max()}'
                                f'Surface min {model.mesh.detach().min()}\n'
                                f'Light Intensity {model.light_intensity.detach()}\n'
                                f'Intensity {model.intensity.detach()}\n'
                                f'X {model.x.detach()}\n'
                                f'Y {model.y.detach()}\n'
                                f'AVG Err {statistics.mean(errs[-10:])}\n'
                                f'Difference lights {torch.linalg.norm(lights.cpu() - model.lights.cpu().detach(), axis=-1)}\n'
                                f'Optimization with lights: {selected_lights}')


                    torch.save(model.rough.detach().cpu(), os.path.join(path2, 'rough.pt'))
                    torch.save(model.diffuse.detach().cpu(), os.path.join(path2, 'diffuse.pt'))
                    torch.save(model.reflectance.detach().cpu(), os.path.join(path2, 'reflectance.pt'))
                    torch.save(model.camera.detach().cpu(), os.path.join(path2, 'camera.pt'))
                    torch.save(model.lights.detach().cpu(), os.path.join(path2, 'lights.pt'))
                    torch.save(model.light_intensity.detach(), os.path.join(path2, 'light_intensity.pt'))
                    torch.save(model.intensity.detach(), os.path.join(path2, 'intensity.pt'))
                    torch.save(model.mesh.detach(), os.path.join(path2, 'surface.pt'))
                    torch.save(model.x.detach(), os.path.join(path2, 'x.pt'))
                    torch.save(model.y.detach(), os.path.join(path2, 'y.pt'))
                    torch.save(model.shadow.detach(), os.path.join(path2, 'shadow.pt'))

                if synthetic:
                    plt.figure(figsize=(20, 10))
                    plt.subplot(1, 2, 1)

                x = np.linspace(0, len(roughs) - 1, len(roughs)) * plot_every
                plt.plot(x, roughs, label='rough', color='red')
                plt.plot(x, diffuses, label='diffuse', color='green')
                plt.plot(x, reflectances, label='reflectance', color='blue')
                # plt.plot(x, intensities, label='intensity')
                plt.xlabel('epoch')
                plt.ylabel('value')
                if synthetic:
                    plt.plot(x, [rough[0]] * len(roughs), color='red', linestyle='dashed')
                    plt.plot(x, [diffuse[0]] * len(diffuses), color='green', linestyle='dashed')
                    plt.plot(x, [reflectance[0]] * len(reflectances), color='blue', linestyle='dashed')
                    plt.title(f'parameters with constant intensity {model.intensity.cpu().detach().numpy().item()}\n'
                              f'Synthetic: (rough,diffuse,reflectance)={(rough[0], diffuse[0], reflectance[0])}; \n'
                              f'Initial value vor prediction: (rough,diffuse,reflectance)={(rough[1], diffuse[1], reflectance[1])}')
                else:
                    plt.title(f'parameters with constant intensity {model.intensity.cpu().detach().numpy().item()}')
                plt.legend()

                if synthetic:
                    plt.subplot(1, 2, 2)

                    surface_line = surface.cpu().detach().numpy()[0, 200, :]
                    pred_surface_line = model.mesh.cpu().detach().numpy()[0, 200, :]

                    x = np.linspace(0, len(surface_line) - 1, len(surface_line))
                    plt.plot(x, surface_line, label='ground truth')
                    plt.plot(x, pred_surface_line, label='prediction')
                    plt.xlabel('x')
                    plt.ylabel('height')
                    plt.legend()

                plt.savefig(os.path.join(path, f'{rough}-{diffuse}-{reflectance}.png'))
                plt.close()

                #plotting_time = TimePlots.event()
                #print(f'Plotting Time {plotting_time}')

    return {'rough' : model.rough.detach().cpu(),
            'diffuse' : model.diffuse.detach().cpu(),
            'reflectance' : model.reflectance.detach().cpu(),
            'camera' : model.camera.detach().cpu(),
            'lights' : model.lights.detach().cpu(),
            'light_intensity': model.light_intensity.detach().cpu(),
            'intensity': model.intensity.detach().cpu(),
            'x': model.x.detach().cpu(),
            'y': model.x.detach().cpu()}