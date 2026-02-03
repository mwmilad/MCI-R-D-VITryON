import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch
import numpy as np

# Assuming your tensor is named 'images'
# images.shape = (16, 1, 256, 2560)

def show_images_grid(images, n_cols=4, figsize=(10, 5)):
    """
    Display images in a grid
    
    Args:
        images: tensor of shape (n_images, 1, H, W)
        n_cols: number of columns in the grid
        figsize: figure size (width, height)
    """
    n_images = images.shape[0]
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for i in range(n_images):
        # Remove channel dimension for display
        img = images[i, 0].cpu().numpy() if torch.is_tensor(images) else images[i, 0]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
