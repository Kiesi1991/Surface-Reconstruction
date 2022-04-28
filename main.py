from optimize_parameters import optimizeParameters
from mainFilament import train_NN

print('1) optimize scene parameters for training')
parameters = optimizeParameters(epochs=1)
print('---'*35)
print('2) train NN with fake images')
train_NN(camera=parameters['camera'],
         lights=parameters['lights'],
         light_intensity=parameters['light_intensity'],
         rough=parameters['rough'],
         diffuse=parameters['diffuse'],
         f0P=parameters['f0P'])