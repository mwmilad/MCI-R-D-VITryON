import os 
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image


class ClothWarpingVVHD(Dataset):
    """
    Custom DataLoader for train Warping Module on VITHON-HD Dataset on data folder.
    """
    def __init__(self, data_path=r'data\zalando-hd-resized\train') -> None:

        # Create Paths
        self.BASE_PATH = data_path
        self.IMAGE_PATH = os.path.join(self.BASE_PATH, 'image') # full body image. (person image)
        self.CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth') # in-shop cloth image.
        self.MASK_CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth-mask') # in-shop cloth full mask image.
        self.DENSE_POSE_PATH = os.path.join(self.BASE_PATH, 'image-densepose')
        self.GT_CLOTH_PATH = os.path.join(self.BASE_PATH, 'gt_cloth_warped_mask') # gt (warped cloth)
        
        # Create list of paths from base path
        self.image_paths = [os.path.join(self.IMAGE_PATH, path) for path in os.listdir(self.IMAGE_PATH)]
        self.cloth_paths = [os.path.join(self.CLOTH_PATH, path) for path in os.listdir(self.CLOTH_PATH)]
        self.mask_cloth_paths = [os.path.join(self.MASK_CLOTH_PATH, path) for path in os.listdir(self.MASK_CLOTH_PATH)]
        self.dense_pose_paths = [os.path.join(self.DENSE_POSE_PATH, path) for path in os.listdir(self.DENSE_POSE_PATH)]
        self.gt_cloth_paths = [os.path.join(self.GT_CLOTH_PATH, path) for path in os.listdir(self.GT_CLOTH_PATH)]

    
    def _test(self):
        return self.image_paths, self.cloth_paths, self.mask_cloth_paths, self.dense_pose_paths, self.gt_cloth_paths
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

dataloader = ClothWarpingVVHD()
out = dataloader._test()
for res in out:
    img = Image.open(res[4156])
    img.show()