import torchvision
import os 
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageDraw
import json
import numpy as np



class ClothWarpingVVHD(Dataset):
    """
    Custom DataLoader for train Warping Module on VITHON-HD Dataset on data folder.
    """

    GRID_PATH = "grid.png"
    def __init__(self, data_path=r'data\zolando-hd-resized\train', w=768, h=1024) -> None:
        self.width, self.height = int(w), int(h)
        
        # Create Paths
        self.BASE_PATH = data_path
        self.IMAGE_PATH = os.path.join(self.BASE_PATH, 'image')
        self.CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth')
        self.MASK_CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth-mask')
        self.DENSE_POSE_PATH = os.path.join(self.BASE_PATH, 'image-densepose')
        self.GT_CLOTH_PATH = os.path.join(self.BASE_PATH, 'gt_cloth_warped_mask')
        
        # Get sorted lists to ensure correspondence
        print(self.IMAGE_PATH)
        self.image_names = sorted(os.listdir(self.IMAGE_PATH))
        self.cloth_names = sorted(os.listdir(self.CLOTH_PATH))
        self.mask_names = sorted(os.listdir(self.MASK_CLOTH_PATH))
        self.gt_names = sorted(os.listdir(self.GT_CLOTH_PATH))
        
        # Create full paths
        self.image_paths = [os.path.join(self.IMAGE_PATH, f) for f in self.image_names]
        self.cloth_paths = [os.path.join(self.CLOTH_PATH, f) for f in self.cloth_names]
        self.mask_cloth_paths = [os.path.join(self.MASK_CLOTH_PATH, f) for f in self.mask_names]
        self.gt_cloth_paths = [os.path.join(self.GT_CLOTH_PATH, f) for f in self.gt_names]
        
        # Define transforms
        self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
                                     ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ---- Read Images ----
        image = Image.open(self.image_paths[idx]).convert('RGB')
        cloth = Image.open(self.cloth_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_cloth_paths[idx]).convert('L')  # Grayscale
        gt = Image.open(self.gt_cloth_paths[idx]).convert('RGB')

        im_name = self.image_names[idx]

        # ---- Load JSON ----
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.BASE_PATH, 'openpose_json', pose_name), 'r') as f:
        
            data = json.load(f)

        keypoints = data["people"][0]["pose_keypoints_2d"]
        keypoints = np.array(keypoints).reshape(-1, 3)

        # ---- Create black mask ----
        im_pose = Image.new("L", (self.width, self.height), 0)  # "L" = grayscale
        pose_draw = ImageDraw.Draw(im_pose)

        # ---- Draw white circles ----
        radius = 10
        i = 0
        pose_map = torch.zeros(keypoints.shape[0], self.height, self.width)
        for x, y, conf in keypoints:
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            if conf > 0.3:  # confidence threshold
                left_up = (x - radius, y - radius)
                right_down = (x + radius, y + radius)
                draw.ellipse([left_up, right_down], fill=255)
                pose_draw.ellipse([left_up, right_down], fill=255)

            one_map = self.mask_transform(one_map)
            pose_map[i] = one_map[0]
            i += 1
            
        
        # Apply transforms
        image = self.img_transform(image) 
        cloth = self.img_transform(cloth) 
        mask = self.mask_transform(mask)
        agnostic = torch.cat([shape, im_h, pose_map], 0)

        gt = self.img_transform(gt) 

        
        return {
            'image': image, # Just for Visualization (Tensor Image)
            'cloth': cloth,
            'mask': mask,
            'im_pose': im_pose, # Just for visualization
            'gt': gt,
            'grid_image': Image.open(self.GRID_PATH), # Just for Visualization
            'image_path': self.image_paths[idx], # Just for Information
            'cloth_path': self.cloth_paths[idx], # Just for Information
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