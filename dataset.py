from torch.utils.data import Dataset
from utils import createSurface
import torch

class DummySet(Dataset):
    def __init__(self, resolution, amount_data=300):
        '''
        intialization of the dataset
        :param resolution: ( tuple ) -> ( H : int , W : int )size of synthetic samples
        :param amount_data: length of dataset
        '''
        self.len = amount_data
        self.resolution = resolution
    def __getitem__(self, index):
        '''
        :param index: (int), index of sample
        :return: synthetic surface sample in pixel-to height representation
        '''
        return torch.tensor(createSurface(self.resolution).tolist()), index
    def __len__(self):
        return self.len