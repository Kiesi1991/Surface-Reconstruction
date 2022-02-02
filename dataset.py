from torch.utils.data import Dataset
from utils import createSurface
import torch

class DummySet(Dataset):
    def __init__(self, resolution, amount_data=100):
        self.len = amount_data
        self.data = torch.tensor([createSurface(resolution).tolist() for i in range(amount_data)], dtype=torch.float64)
    def __getitem__(self, index):
        return self.data[index], index
    def __len__(self):
        return self.len