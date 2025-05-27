import torch
import numpy as np
import h5py
import cv2 as cv
import os
from monai.networks.nets import SwinUNETR, UNETR, UNet, VNet, BasicUNetPlusPlus
import time
import argparse

from utils.utils import *
from nets.swin_ddf import SwinDDF

parser = argparse.ArgumentParser(description="param of training")
parser.add_argument('--images_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--output_fname', type=str, default=None)
args = parser.parse_args()

images_dir = args.images_dir
output_dir = args.output_dir
output_fname = args.output_fname

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = {
    'model':'SwinDDF',
    'ckpt_path':f'/myproject_outputs/710/best_dice_weights.pth'
}
ckpt_path = cfg['ckpt_path']
model_name = cfg['model']

print(f"[INFO] Config:{cfg}")

if model_name == 'SwinUNETR':
    model = SwinUNETR(
            img_size=(128,128,128),
            in_channels=1,
            out_channels=2,
            use_v2 = True
        ).to(device)
elif model_name == 'SwinDDF':
    model = SwinDDF(
            img_size=(128,128,128),
            in_channels=1,
            out_channels=2,
        ).to(device)
elif model_name == 'UNet':
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(4, 8, 16, 32, 64),
        strides=(2, 2, 2, 2),
    ).to(device)
elif model_name == 'UNETR':
    model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(128, 128, 128),
    ).to(device)
elif model_name == 'UNetPP':
    model = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        deep_supervision=True
    ).to(device)


if model_name == 'UNetPP':
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
print(f"[INFO] Model loaded from {ckpt_path}")

patch_size = (128, 128, 128)
step = 96

print("Loading images...")
images = read_images_to_volume(images_dir)

mask = sliding_window_infer(model, images / 255, patch_size, step)

print("saving result...")
save_h5(mask, os.path.join(output_dir, output_fname))

