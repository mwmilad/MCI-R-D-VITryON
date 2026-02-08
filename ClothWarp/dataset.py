import os 
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class ClothWarpingVVHD(Dataset):
    """
    Custom DataLoader for train Warping Module on VITHON-HD Dataset on data folder.
    """
    def __init__(self, data_path=r'ClothWarp\data\zalando-hd-resized\train') -> None:
        self.BASE_PATH = data_path
        self. IMAGE_PATH = os.path.join(self.BASE_PATH, 'image')
        print(self.IMAGE_PATH)

    
    def _test(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

dataloader = ClothWarpingVVHD()