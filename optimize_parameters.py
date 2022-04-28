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

def optimizeParameters(path_target='realSamples', path_results='results', lr = 1e-3, epochs = 3001):
    samples = get_real_samples(path_target)
    camera, lights, mesh = get_scene_locations(batch_real_samples=samples.shape[0])
    path = create_next_folder(path_results)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = OptimizeParameters(mesh, lights, camera, path=None, device=device)

    parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(name)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    errs = []
    errors = []
    for epoch in range(epochs):
            pred = model.forward()
            #err = mse(pred[:,:,4:], samples[:,:,4:])
            err = mse(pred.to(device), samples.to(device)) #+ 0.001 * (model.mesh**2).sum()
            errs.append(err.item())
            optimizer.zero_grad()
            err.backward()
            #torch.nn.utils.clip_grad_value_(model.mesh, 0.001)
            optimizer.step()
            if epoch % 10 == 0:
                # print(f'Epoch {epoch} AVG Err {statistics.mean(errs[-10:])} Surface Max {model.mesh.detach().max()} Surface Min {model.mesh.detach().min()}')
                errors.append(statistics.mean(errs[-10:]))
            if epoch % 100 == 0:
                os.mkdir(os.path.join(path, f'Epoch-{epoch}'))
                path2 = os.path.join(path, f'Epoch-{epoch}')
                print(f'Rough {torch.sigmoid(model.rough).item()} Diffuse {torch.sigmoid(model.diffuse).item()} f0P {torch.sigmoid(model.f0P).item()}')
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

                with open(os.path.join(path2, 'parameters.txt'), 'w') as f:
                    f.write(f'Parameters {parameters}\n'
                            f'Rough {torch.sigmoid(model.rough).item()} Diffuse {torch.sigmoid(model.diffuse).item()} f0P {torch.sigmoid(model.f0P).item()} \n'
                            f'Camera {model.camera.detach()}\n'
                            f'Lights {model.lights.detach()}\n'
                            f'Surface {model.mesh.detach()}\n'
                            f'Surface Max {model.mesh.detach().max()}'
                            f'Surface min {model.mesh.detach().min()}\n'
                            f'Light Intensity {model.light_intensity.detach()}\n'
                            f'Light Color {model.light_color.detach()}\n'
                            f'X {model.x.detach()}\n'
                            f'Y {model.y.detach()}\n'
                            f'AVG Err {statistics.mean(errs[-10:])}')

                torch.save(model.rough.detach().cpu(), os.path.join(path2, 'rough.pt'))
                torch.save(model.diffuse.detach().cpu(), os.path.join(path2, 'diffuse.pt'))
                torch.save(model.f0P.detach().cpu(), os.path.join(path2, 'f0P.pt'))
                torch.save(model.camera.detach().cpu(), os.path.join(path2, 'camera.pt'))
                torch.save(model.lights.detach().cpu(), os.path.join(path2, 'lights.pt'))
                torch.save(model.light_intensity.detach(), os.path.join(path2, 'light_intensity.pt'))
                torch.save(model.mesh.detach(), os.path.join(path2, 'surface.pt'))

    return {'rough' : model.rough.detach().cpu(),
            'diffuse' : model.diffuse.detach().cpu(),
            'f0P' : model.f0P.detach().cpu(),
            'camera' : model.camera.detach().cpu(),
            'lights' : model.lights.detach().cpu(),
            'light_intensity': model.light_intensity.detach().cpu()}