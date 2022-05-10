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

def optimizeParameters(path_target='realSamples', path_results=os.path.join('results', 'optimization'), lr = 1e-3, epochs = 1):
    samples = get_real_samples(path_target)
    camera, lights, mesh = get_scene_locations(batch_real_samples=samples.shape[0])
    path = create_next_folder(path_results)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = OptimizeParameters((mesh,True), (lights,False), (camera,False), device=device)

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
    for epoch in range(epochs):
            pred = model.forward()
            #err = mse(pred[:,:,4:], samples[:,:,4:])
            err = mse(pred.to(device), samples.to(device)) #+ 0.001 * (model.mesh**2).sum()
            errs.append(err.item())
            optimizer.zero_grad()
            err.backward()
            #torch.nn.utils.clip_grad_value_(model.mesh, 0.001)
            optimizer.step()
            if epoch==epochs//2:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr/10, weight_decay=0.0)
                print(f'New Lr {lr/10}')
            if epoch % 50 == 0:
                # print(f'Epoch {epoch} AVG Err {statistics.mean(errs[-10:])} Surface Max {model.mesh.detach().max()} Surface Min {model.mesh.detach().min()}')
                errors.append(statistics.mean(errs[-10:]))
                print(f'Epoch {epoch} / {epochs}')
            if epoch % 1000 == 0:
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

                roughs.append(torch.sigmoid(model.rough).cpu().detach().numpy().item())
                diffuses.append(torch.sigmoid(model.diffuse).cpu().detach().numpy().item())
                f0s.append(torch.sigmoid(model.f0P).cpu().detach().numpy().item())
                intensities.append(model.intensity.cpu().detach().numpy().item())

                x = np.linspace(0, len(roughs) - 1, len(roughs))
                plt.plot(x, roughs, label='rough')
                plt.plot(x, diffuses, label='diffuse')
                plt.plot(x, f0s, label='f0')
                plt.plot(x, intensities, label='intensity')
                plt.xlabel('epoch')
                plt.ylabel('value')
                plt.legend()
                plt.savefig(os.path.join(path, f'parameters.png'))
                plt.close()

                with open(os.path.join(path2, 'parameters.txt'), 'w') as f:
                    f.write(f'Parameters {parameters}\n'
                            f'Rough {torch.sigmoid(model.rough).item()} Diffuse {torch.sigmoid(model.diffuse).item()} f0P {torch.sigmoid(model.f0P).item()} \n'
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

    return {'rough' : model.rough.detach().cpu(),
            'diffuse' : model.diffuse.detach().cpu(),
            'f0P' : model.f0P.detach().cpu(),
            'camera' : model.camera.detach().cpu(),
            'lights' : model.lights.detach().cpu(),
            'light_intensity': model.light_intensity.detach().cpu(),
            'intensity': model.intensity.detach().cpu(),
            'x': model.x.detach().cpu(),
            'y': model.x.detach().cpu()}