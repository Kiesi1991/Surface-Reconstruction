from torch import nn

class zPrediction(nn.Module):
    def __init__(self):
        super(zPrediction, self).__init__()

    def forward(self, x):
        return x