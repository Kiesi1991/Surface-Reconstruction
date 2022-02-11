from torch import nn

class zPrediction(nn.Module):
    def __init__(self):
        super(zPrediction, self).__init__()
        self.conv1 = nn.Conv2d(12, 6, kernel_size=5, padding=5 // 2)
        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv2d(3, 1, kernel_size=3, padding=3 // 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x.squeeze(1)