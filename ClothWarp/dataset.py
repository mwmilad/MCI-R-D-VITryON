import os 
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class ClothWarpingVVHD(Dataset):
    def __init__(self, data_path) -> None:
        self.BASE_PATH = data_path
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
