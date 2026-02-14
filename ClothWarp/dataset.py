import torchvision
import os 
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class ClothWarpingVVHD(Dataset):
    """
    Custom DataLoader for train Warping Module on VITHON-HD Dataset on data folder.
    """

    GRID_PATH = "grid.png"
    def __init__(self, data_path=r'data\zolando-hd-resized\train', w=768, h=1024) -> None:
        self.w, self.h = int(w / 2), int(h / 2)
        
        # Create Paths
        self.BASE_PATH = data_path
        self.IMAGE_PATH = os.path.join(self.BASE_PATH, 'image')
        self.CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth')
        self.MASK_CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth-mask')
        self.DENSE_POSE_PATH = os.path.join(self.BASE_PATH, 'image-densepose')
        self.GT_CLOTH_PATH = os.path.join(self.BASE_PATH, 'gt_cloth_warped_mask')
        
        # Get sorted lists to ensure correspondence
        self.image_files = sorted(os.listdir(self.IMAGE_PATH))
        self.cloth_files = sorted(os.listdir(self.CLOTH_PATH))
        self.mask_files = sorted(os.listdir(self.MASK_CLOTH_PATH))
        self.gt_files = sorted(os.listdir(self.GT_CLOTH_PATH))
        
        # Create full paths
        self.image_paths = [os.path.join(self.IMAGE_PATH, f) for f in self.image_files]
        self.cloth_paths = [os.path.join(self.CLOTH_PATH, f) for f in self.cloth_files]
        self.mask_cloth_paths = [os.path.join(self.MASK_CLOTH_PATH, f) for f in self.mask_files]
        self.gt_cloth_paths = [os.path.join(self.GT_CLOTH_PATH, f) for f in self.gt_files]
        
        # Define transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read images
        image = Image.open(self.image_paths[idx]).convert('RGB')
        cloth = Image.open(self.cloth_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_cloth_paths[idx]).convert('L')  # Grayscale
        gt = Image.open(self.gt_cloth_paths[idx]).convert('RGB')
        
        # Apply transforms
        image = self.img_transform(image)
        cloth = self.img_transform(cloth)
        mask = self.mask_transform(mask)
        gt = self.img_transform(gt)
        
        return {
            'image': image,
            'cloth': cloth,
            'mask': mask,
            'gt': gt,
            'grid_image': Image.open(self.GRID_PATH),
            'image_path': self.image_paths[idx],
            'cloth_path': self.cloth_paths[idx],
        }

# Test the dataset
if __name__ == "__main__":
    dataset = ClothWarpingVVHD()
    
    # Get a sample
    sample = dataset[0]
    sample['grid_image'].show()
    
    print(f"Image shape: {sample['image'].shape}")
    print(f"Cloth shape: {sample['cloth'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"GT shape: {sample['gt'].shape}")
    print(f"Image min/max: {sample['image'].min():.3f}/{sample['image'].max():.3f}")
    print(f"Mask min/max: {sample['mask'].min():.3f}/{sample['mask'].max():.3f}")