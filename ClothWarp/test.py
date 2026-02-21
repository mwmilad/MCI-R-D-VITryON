#coding=utf-8
# cp viton code base
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time

from networks import GMM
from dataset import ClothWarpingVVHD

from typing import Literal

#from networks import GMM UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

# from tensorboardX import SummaryWriter
# from visualization import board_add_image, board_add_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

device: Literal['cpu', 'cuda'] = 'cuda' if torch.cuda.is_available() else 'cpu'

opt = get_opt()
model = GMM(opt)
count_params = sum(p.numel() for p in model.parameters())
print(f"GMM Network Create.Number of Parameter is : {count_params / 1_000_000} M")

train_loader = ClothWarpingVVHD(r'.\data')
print(f"data Network Create.Number of data is : {len(train_loader)}")

model.to(device)
model.train()

# criterion
criterionL1 = nn.L1Loss()

batch = next(iter(train_loader))
print(type(batch))


# # optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
#         max(0, step - opt.keep_step) / float(opt.decay_step + 1))

# for step in range(opt.keep_step + opt.decay_step):
#     iter_start_time = time.time()
#     inputs = next(iter(train_loader))
#     image = inputs['image'].to(device)
#     print(op)