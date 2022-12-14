from torch.utils.data import DataLoader
from dataset import DummySet
from shaders import FilamentShading
from models import *
from tqdm import tqdm

# resolution of images
resolution = (386, 516)
# path, where results are stored
path_results = os.path.join('results', 'trainNN20')
# path, where optimized parameters are stored
path_optimized_parameters = os.path.join('results', 'optimization', '1', '0')
# path, where real images samples are stored
path_real_samples = 'realSamples1'

# please enter parameters for SurfaceNet
# if more than 1 element in list, the training is looped accordingly
mid_channels=[64] # mid_channels C
layers=[8] # amount of BlockNet layers
blocks = [ConvBlock] # BlockNet: [ResNextBlock, ConvBlock]
cardinality = [1] # cardinality for ResNextBlock

# training parameters
num_iter = 2001 # amount of synthetic surface samples during training
batch_size = 2 # batchsize
lr = 1e-4 # learining rate
crop = 50 # all images will be cropped accordingly
encoder_decoder = [False] # use encoder decoder training loop
sigma = [0.98] # if sigma is 1 full surface loss

# loop over parameters in lists
for _layers in layers:
    for _mid_channels in mid_channels:
        for block in blocks:
            for c in cardinality:
                for _sigma in sigma:
                    for _encoder_decoder in encoder_decoder:
                        torch.cuda.empty_cache()
                        # if BlockNet is not ResNextBlock, cardinality must be 1
                        if block != ResNextBlock and c != 1:
                            break
                        # get real cabin-cap image
                        real_samples = getRealSamples(path_real_samples)
                        # create folder, where intermediate/final results are stored
                        path = createNextFolder(path_results)
                        # save parameters in textfile
                        with open(os.path.join(path, 'parameters.txt'), 'w') as f:
                            f.write(f'{str(block)}\n'
                                    f'l = {_layers}\n'
                                    f'CM = {_mid_channels}\n'
                                    f'c = {c}\n'
                                    f'sigma = {_sigma}\n'
                                    f'enc_dec = {_encoder_decoder}')
                        # initialization of SurfaceNet
                        model = SurfaceNet(layers=_layers, mid_channels=_mid_channels, BlockNet=block, cardinality=c)
                        # check if cuda is available
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        # model to device
                        model.to(device)
                        try:
                            # get optimized scene parameters from step 1
                            optimized_parameters = getOptimizedParameters(path_optimized_parameters)
                        except:
                            raise ('Please enter a folder path including optimized parameters')
                        # create transformation with optimized scene parameters
                        transformation = FilamentShading(optimized_parameters)
                        # initialization of dataset with synthetic surfaces
                        dataset = DummySet(resolution, amount_data=num_iter)
                        # create loader for training
                        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                        ############################################################################
                        # Update network
                        ############################################################################

                        def _forward(network: nn.Module, data: DataLoader, metric: callable):
                            device = next(network.parameters()).device
                            for synthetic_surface, idx in data:
                                # transform synthetic surface to synthetic images
                                synthetic_images = transformation.forward(synthetic_surface.to(device)) # (B,L,H,W)
                                # predict surface with synthetic images as input
                                predicted_surface = model(synthetic_images) # (B,H,W)
                                # calculate loss between synthetic surface and predicted surface
                                surface_err = metric(predicted_surface[..., crop:-crop, crop:-crop],
                                                                (synthetic_surface[..., crop:-crop, crop:-crop]).to(device))
                                if _encoder_decoder:
                                    # apply decoder: transform prediceted surface to predicted images
                                    predicted_images = transformation.forward(predicted_surface)
                                    # calculate loss between synthetic images and predicted images
                                    res = metric(predicted_images[..., crop:-crop, crop:-crop],
                                                 synthetic_images[..., crop:-crop, crop:-crop])
                                else:
                                    # calculate gradients for predicted surface
                                    predicted_gradients = getGradients(predicted_surface[..., crop:-crop, crop:-crop])
                                    # calculate gradients for synthetic surface
                                    synthetic_gradients = getGradients((synthetic_surface[..., crop:-crop, crop:-crop]).to(device))
                                    # calculate gradient loss
                                    gradients_err = metric(predicted_gradients, synthetic_gradients)
                                    # calculate total loss
                                    res = (1-_sigma) * gradients_err + _sigma * surface_err
                                yield res, surface_err.item()

                        @torch.enable_grad()
                        def update(network: nn.Module, data: DataLoader, loss: nn.Module,
                                   opt: torch.optim.Optimizer) -> list:
                            # set network to train mode
                            network.train()
                            # initialize list for storing training error
                            errs, surface_errs, norm_grads = [], [], []
                            for iter, (err, surface_err) in tqdm(enumerate(_forward(network, data, loss))):
                                # store training error
                                errs.append(err.item())
                                surface_errs.append(surface_err)
                                # set gradients to zero, apply backward pass and perform optimization step
                                opt.zero_grad()
                                (err).backward()
                                if len(norm_grads) <= 10:
                                    norm_grads.append(torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5).item())
                                else:
                                    norm_grads.append(torch.nn.utils.clip_grad_norm_(network.parameters(),
                                                                                        max_norm=mean(norm_grads[(len(norm_grads)//2):])).item())
                                opt.step()
                                # every 100 loops apply some plotting function and store intermediate/final results
                                if (iter % 100) == 0 and iter != 0:
                                    # create folder for storing results
                                    path1 = createNextFolder(path)
                                    # predict surface for real sample
                                    predicted_surface = model(real_samples.to(device).permute(0,4,2,3,1).squeeze(-1))  # (B,H,W)
                                    # transform predicted surface to predicted images
                                    predicted_images = transformation.forward((predicted_surface))  # (B,L,H,W)
                                    # plot image comparison
                                    network.plotImageComparism(real_samples, predicted_images, path1)
                                    # plot height profiles
                                    network.plotProfileDiagrams(transformation.optimized_surface, predicted_surface, path1)
                                    # plot training errors
                                    network.plotErrorDiagram(errs, path)
                                    # plot training surface errors
                                    network.plotErrorDiagram(surface_errs, path, name='surface error')
                                    # plot gradient norms
                                    network.plotErrorDiagram(norm_grads, path, name='gradient norms')
                                    # save model
                                    torch.save(model.state_dict(), os.path.join(path1, 'model.pt'))
                            return errs

                        ############################################################################
                        # training and evaluation
                        ############################################################################

                        mse = torch.nn.MSELoss() # MSE loss function
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0) # Adam optimizer
                        # perform update function
                        errs = update(model, trainloader, mse, optimizer)