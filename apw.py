import torch
import torchvision
from ClothWarp.dataset import HDVitonDataset, HDVitonDataLoader
from u_net import UNet
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have:
# image_tensor: shape [C, H, W] or [1, C, H, W] (values typically 0-1 or 0-255)
# mask_tensor: shape [H, W] or [1, H, W] (values 0 or 1, or 0-255)

def apply_mask_and_show(image_tensor, mask_tensor):
    """
    Apply mask to image and display result
    """
    
    # Ensure tensors are on CPU and convert to numpy if needed
    if torch.is_tensor(image_tensor):
        image = image_tensor.detach().cpu()
    else:
        image = torch.tensor(image_tensor)
    
    if torch.is_tensor(mask_tensor):
        mask = mask_tensor.detach().cpu()
    else:
        mask = torch.tensor(mask_tensor)
    
    # Handle different tensor shapes
    # Remove batch dimension if present
    if image.dim() == 4:  # [B, C, H, W]
        image = image[0]
    if mask.dim() == 3:  # [1, H, W] or [B, H, W]
        mask = mask.squeeze()
    
    # Ensure mask is 2D [H, W]
    if mask.dim() == 2:
        # Add channel dimension for broadcasting
        mask = mask.unsqueeze(0)  # [1, H, W]
    
    # Normalize mask to 0-1 if it's not already
    if mask.max() > 1:
        mask = mask / 255.0
    
    # Apply mask (broadcasting over channels)
    masked_image = image * mask  # [C, H, W]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy for display
    # Image tensor: from [C, H, W] to [H, W, C]
    img_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()
    masked_np = masked_image.squeeze(0).permute(1, 2, 0).numpy()
    
    # Clip values to valid range [0, 1] for display
    img_np = np.clip(img_np, 0, 1)
    masked_np = np.clip(masked_np, 0, 1)
    
    # Display original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display mask
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    # Display masked image
    axes[2].imshow(masked_np)
    axes[2].set_title('Masked Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return masked_image

# Example usage
# masked_result = apply_mask_and_show(image_tensor, mask_tensor)

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", default = "data\zolando-hd-resized")
parser.add_argument("--datamode", default = "train")
parser.add_argument("--stage", default = "GMM")
parser.add_argument("--data_list", default = "train_pairs.txt")
parser.add_argument("--fine_width", type=int, default = 768)
parser.add_argument("--fine_height", type=int, default = 1024)
parser.add_argument("--radius", type=int, default = 5)
parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-j', '--workers', type=int, default=0)
opt = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = HDVitonDataset(opt=opt)
train_loader = HDVitonDataLoader(opt, train_data)

first_item = train_data.__getitem__(0)
first_batch = train_loader.next_batch()

model = UNet(n_channels=6).to(device)

# output = model(torch.rand(16, 6, 512, 384).to(device))

print('apply')
apply_mask_and_show(image_tensor=first_batch['image'], mask_tensor=first_batch['agnostic_mask'])

