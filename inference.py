import torch
import numpy as np
import h5py
import cv2 as cv
import os
from monai.networks.nets import SwinUNETR, UNETR, UNet, VNet, BasicUNetPlusPlus
import time
import argparse

from utils.utils import load_tiff, save_h5, sliding_window_infer, read_images_to_volume
from nets.swin_dfn import SwinDFNet

parser = argparse.ArgumentParser(description="param of training")
parser.add_argument('--range', type=str, default=None)
parser.add_argument('--region', type=int, default=None)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# task_id = 228
cfg = {
    'model':'FFSwinUNETR',
    # 'ckpt_path':f'/data/XieHao/128nm_EM/ckpt/{task_id}/best_dice_weights.pth'
    # 'ckpt_path':f'./ckpt/ffswin_693_ns12.pth'
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
        ).to(device)
elif model_name == 'FFSwinUNETR':
    model = SwinDFNet(
            img_size=(128,128,128),
            in_channels=1,
            out_channels=2,
            net_setting = 12
        ).to(device)
elif model_name == 'UNet':
    # model = UNet(
    #         spatial_dims=3,
    #         in_channels=1,
    #         out_channels=2,
    #         channels=(16, 32, 64, 128, 256),
    #         strides=(2, 2, 2, 2),
    #         num_res_units=2,
    # ).to(device)
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
        # feature_size=24,
        # hidden_size=1024,
        # mlp_dim=3072,
        # num_heads=16,
        # pos_embed="conv",
        # norm_name="instance",
        # res_block=True,
        # dropout_rate=0.1,
    ).to(device)
elif model_name == 'UNetPP':
    model = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        deep_supervision=True
    ).to(device)
# model = VNet(
#         in_channels=1,
#         out_channels=2,
#     ).to(device)


if model_name == 'UNetPP':
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    # 移除state_dict中的module.前缀
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
print(f"[INFO] Model loaded from {ckpt_path}")

if args.range == '1-4':
    images_dirs = ['mec_128nm1', 'mec_128nm2', 'mec_128nm3', 'mec_128nm4']
elif args.range == '5-8':
    images_dirs = ['mec_128nm5', 'mec_128nm6', 'mec_128nm7', 'mec_128nm8']
elif args.range == '9-12':
    images_dirs = ['mec_128nm9', 'mec_128nm10', 'mec_128nm11', 'mec_128nm12']
elif args.range == '13-16':
    images_dirs = ['mec_128nm13', 'mec_128nm14', 'mec_128nm15', 'mec_128nm16']
elif args.range == '17-20':
    images_dirs = ['mec_128nm17', 'mec_128nm18', 'mec_128nm19', 'mec_128nm20']

region = args.region

output_dir = os.path.join('/data/XieHao/128nm_EM/mec_128nm_seg_and_analysis', f'{args.range}_region{region}')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

image_paths = []
for images_dir in images_dirs:
    images_fnames = sorted(os.listdir(f'/data/XieHao/128nm_EM/mec_128nm/{images_dir}'))
    for fname in images_fnames:
        image_path = f'/data/XieHao/128nm_EM/mec_128nm/{images_dir}/{fname}'
        image_paths.append(image_path)
num_images = len(image_paths)
print(f"num of images: {num_images}")

h, w = cv.imread(image_paths[0], cv.IMREAD_GRAYSCALE).shape
images = []
for i, image_path in enumerate(image_paths):
    print(f"loading image {i+1}/{num_images}, {image_path}")
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if region == 1:
        image = image[:h//2, :w//2]
    elif region == 2:
        image = image[:h//2, w//2:]
    elif region == 3:
        image = image[h//2:, :w//2]
    elif region == 4:
        image = image[h//2:, w//2:]
    images.append(image)
images = np.array(images)
print("Saving images...")
# save_h5(images, f'../output/mec{range}_region{region}_images.h5')
save_h5(images, os.path.join(output_dir, f'mec{args.range}_region{region}_images.h5'))

def save_volume_to_images(volume, images_dir):
    for index, image in enumerate(volume):
        fname = str(index+1).zfill(4) + '.png'
        cv.imwrite(os.path.join(images_dir, fname), image * 255)


patch_size = (128, 128, 128)
step = 96

# block3 = load_tiff('/data/XieHao/128nm_EM/tiff/block3.tiff') / 255
# images_dir = '/data/XieHao/128nm_EM/mec_128nm/mec_128nm1'
# images_dir = args.images_dir
# print("Loading images...")
# block3 = read_images_to_volume('./data/block3_image')
# print("Images loaded.")
# block4 = load_tiff('./data/block4.tiff') / 255

mask = sliding_window_infer(model, images / 255, patch_size, step, use_tta=False, deep_supervision=True, out_depth_index=0)
# block4_mask = sliding_window_infer(model, block4, patch_size, step)
print("saving result...")
# save_h5(block3_mask, f"/data/XieHao/128nm_EM/ckpt/{task_id}/block3_best_signedmap_contour2{model_name}.h5")
# save_h5(block3_mask, f"/data/XieHao/128nm_EM/mec_128nm_seg/mec_128nm1.h5")
# save_h5(block3_mask, args.output_seg_path)
# print("result saved.")
# print("saving images...")
# save_h5(block3, args.output_image_path)
# save_h5(mask, f'../output/mec5-8_region{region}_seg.h5')
save_h5(mask, os.path.join(output_dir, f'mec{args.range}_region{region}_seg.h5'))



'''
python inference.py --images_dir /data/XieHao/128nm_EM/mec_128nm/mec_128nm1 --output_seg_path /data/XieHao/128nm_EM/mec_128nm_seg/mec_128nm1.h5 --output_image_path /data/XieHao/128nm_EM/mec_128nm_seg/mec_128nm1.h5
'''