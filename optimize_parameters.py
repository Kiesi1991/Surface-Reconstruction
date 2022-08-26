from models import OptimizeParameters
import matplotlib.pyplot as plt
import cv2
import statistics
from utils import *
from tqdm import tqdm
from shaders import FilamentShading

def optimizeParameters(path_target='realSamples', path_results=os.path.join('results', 'optimization'),
                       lr=1e-4, weight_decay=0.,
                       iterations=3001, mean_intensity=1.0,
                       intensity=2.5, selected_lights='all', para_lights=True,
                       rough=(0.5,0.5,True), diffuse=(0.5,0.5,True), reflectance=(0.5,0.5,True),
                       synthetic=False, surface_opimization=True, quick_search=False,
                       regularization_function='exp'):

    plot_every = iterations//8

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(os.path.join(path_results)):
        os.mkdir(os.path.join(path_results))

    samples = getRealSamples(path_target)

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
    path = path_results if quick_search else createNextFolder(path_results)

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

    model.lights.requires_grad = False

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
    l_to_origin = []
    l_to_zero = []
    lam = 0.000001
    active_lights = abs(end - start) + 1

    for iteration in tqdm(range(iterations)):
            pred = model.forward()
            distance = torch.linalg.norm(lights.to(device) - model.lights, axis=-1)
            if regularization_function == 'exp':
                distance_err = torch.exp(torch.sum(distance, dim=-1))/active_lights
            elif regularization_function == 'square':
                distance_err = (torch.sum(distance, dim=-1) ** 2) / active_lights
            else:
                raise(f'Regularisation function is not defined: {regularization_function}!')

            err = mse(pred[...,start:end+1].to(device), samples[...,start:end+1].to(device)) + lam * distance_err
            errs.append(err.item())
            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if iteration == plot_every and para_lights:
                model.lights.requires_grad = True

            if iteration % 50 == 0:
                errors.append(statistics.mean(errs[-10:]))

            if iteration % plot_every == 0:

                roughs.append(model.rough.cpu().detach().numpy().item())
                diffuses.append(model.diffuse.cpu().detach().numpy().item())
                reflectances.append(model.reflectance.cpu().detach().numpy().item())
                intensities.append(model.intensity.cpu().detach().numpy().item())

                if not quick_search:
                    os.mkdir(os.path.join(path, f'iteration-{iteration}'))
                    path2 = os.path.join(path, f'iteration-{iteration}')
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

                    l_to_origin.append(torch.linalg.norm(lights.cpu().detach() - model.lights.cpu().detach(), axis=-1).tolist())
                    x = np.linspace(0, len(l_to_origin) - 1, len(l_to_origin)) * plot_every

                    for L in range(12):
                        plt.plot(x, np.array(l_to_origin)[:,L], label=f'{L}')

                    plt.xlabel('iteration')
                    plt.ylabel('distance to origin')
                    plt.legend()
                    plt.savefig(os.path.join(path, 'l_to_origin.png'))
                    plt.close()

                    l_to_zero.append(torch.linalg.norm(model.lights.cpu().detach(), axis=-1).tolist())
                    x = np.linspace(0, len(l_to_zero) - 1, len(l_to_zero)) * plot_every

                    for L in range(12):
                        plt.plot(x, np.array(l_to_zero)[:, L], label=f'{L}')

                    plt.xlabel('iteration')
                    plt.ylabel('distance to zero')
                    plt.yscale('log')
                    plt.legend()
                    plt.savefig(os.path.join(path, 'l_to_zero.png'))
                    plt.close()

                    x = np.linspace(0, len(errors) - 1, len(errors)) * plot_every
                    plt.plot(x, errors, label='errors')
                    plt.xlabel('iteration')
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

                    normal_vectors = getNormals(model.mesh.detach())
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
                                f'Distance Err {distance_err * lam}\n'
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

                plt.xlabel('iterations')
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

    return {'rough' : model.rough.detach().cpu(),
            'diffuse' : model.diffuse.detach().cpu(),
            'reflectance' : model.reflectance.detach().cpu(),
            'camera' : model.camera.detach().cpu(),
            'lights' : model.lights.detach().cpu(),
            'light_intensity': model.light_intensity.detach().cpu(),
            'intensity': model.intensity.detach().cpu(),
            'x': model.x.detach().cpu(),
            'y': model.x.detach().cpu()}