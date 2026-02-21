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
        self.width, self.height = int(w / 2), int(h / 2)
        
        # Create Paths
        self.BASE_PATH = data_path
        self.IMAGE_PATH = os.path.join(self.BASE_PATH, 'image')
        self.CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth')
        self.MASK_CLOTH_PATH = os.path.join(self.BASE_PATH, 'cloth-mask')
        self.DENSE_POSE_PATH = os.path.join(self.BASE_PATH, 'image-densepose')
        self.GT_CLOTH_PATH = os.path.join(self.BASE_PATH, 'gt_cloth_warped_mask')
        
        # Get sorted lists to ensure correspondence
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
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


        im_name = self.image_names[idx]
                # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(os.path.join(self.BASE_PATH, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))
        
        r = 5
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        im_pose = Image.new('L', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]
        
        # Apply transforms
        image = self.img_transform(image) 
        cloth = self.img_transform(cloth) 
        mask = self.mask_transform(mask)
        im_pose = self.img_transform(im_pose)
        gt = self.img_transform(gt) 

        
        return {
            'image': image, # Just for Visualization
            'cloth': cloth,
            'mask': mask,
            'im_pose': im_pose
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