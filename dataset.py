from torch.utils.data import Dataset
from utils import createSurface
import torch

class DummySet(Dataset):
    def __init__(self, resolution, para, amount_data=300):
        self.len = amount_data
        self.resolution = resolution
        self.para = para
    def __getitem__(self, index):
        return torch.tensor(createSurface(self.resolution, para=self.para).tolist()), index
    def __len__(self):
        return self.len