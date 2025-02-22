import os
import time

import torch
import h5py
import cv2 as cv
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import torch.nn.functional as F
from tqdm import tqdm
from scipy import ndimage
from patchify import patchify
from monai.transforms.utils import distance_transform_edt
from monai.transforms import (
    Flip,
    Rotate90,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

def load_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def save_nii(arr, path):
    sitk.WriteImage(sitk.GetImageFromArray(arr), path)

def load_h5(path):
    with h5py.File(path, 'r') as fl:
        return np.array(fl['main'])

def save_h5(data, path):
    with h5py.File(path, 'w') as hdf:
        hdf.create_dataset('main', data=data, compression="gzip", compression_opts=9)
        # print(f"hdf file has saved to {path}")

def load_tiff(path):
    return tiff.imread(path)

def save_images_as_tiff(images, output_path):
    tiff.imwrite(output_path, np.array(images), photometric='minisblack')

def read_images_to_volume(images_dir):
    images = [cv.imread(os.path.join(images_dir,f), 0) for f in os.listdir(images_dir) if f.endswith('.png')]
    return np.array(images)

def save_volume_to_images(volume, images_dir):
    flag = volume.max() > 100 # 判断volume的范围，是0-1还是0-255
    for index, image in enumerate(volume):
        fname = str(index+1).zfill(4) + '.png'
        if flag:
            cv.imwrite(os.path.join(images_dir, fname), image)
        else:
            cv.imwrite(os.path.join(images_dir, fname), image * 255)

def infer_patch(model, x, deep_supervision=True, out_depth_index=0):
    '''
    input: x: Tensor(1, d, h, w), 可以不用将x移到device中，函数自动移到模型所在GPU
    return: y: Tensor(num_cls, d, h, w) soft label
    '''
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    x = x.unsqueeze(0).float().to(device) # (1, 1, d, h, w)
    if deep_supervision:
        y_pred = F.softmax(model(x)[out_depth_index][0], dim=0) # (num_cls, d, h, w)
    else:
        y_pred = F.softmax(model(x)[0], dim=0) # (num_cls, d, h, w)
    # y_pred = F.softmax(model(x)[0][:2, :, :, :], dim=0) # (num_cls, d, h, w)
    return y_pred

def infer_with_tta(model, x, tta_transforms=None):
    '''
    input: x: Tensor(1, d, h, w), 可以不用将x移到device中，函数自动移到模型所在GPU
    return: y: Tensor(num_cls, d, h, w) soft label
    '''
    if tta_transforms is None:
        # 定义 TTA 操作
        tta_transforms = [
            lambda x: x,  # 原始图像
            Flip(spatial_axis=[0]),         # x轴翻转
            Flip(spatial_axis=[1]),         # y轴翻转
            Flip(spatial_axis=[2]),         # z轴翻转
            Rotate90(k=1, spatial_axes=[0, 1]),  # 90度旋转 (x, y)
            Rotate90(k=1, spatial_axes=[0, 2]),  # 90度旋转 (x, z)
            Rotate90(k=1, spatial_axes=[1, 2]),  # 90度旋转 (y, z)
        ]
    num_cls = 2
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    y_pred_tta = torch.zeros([num_cls, 128, 128, 128]).to(device)
    for tta_transform in tta_transforms:
        x_tta = tta_transform(x).unsqueeze(0).float().to(device) # (1, 1, d, h, w)
        y_pred = F.softmax(model(x_tta)[0], dim=0) # (num_cls, d, h, w)
        # 对预测结果进行逆变换，以恢复到原始方向
        if hasattr(tta_transform, "inverse"):
            y_pred = tta_transform.inverse(y_pred)
        # 累加每次 TTA 的预测结果
        y_pred_tta += y_pred
    # 计算平均 TTA 结果
    y_pred_tta /= len(tta_transforms)
    return y_pred_tta

def refine_volume_xy(refiner, raw_images, pred_masks, show_progress=False):
    '''
        raw_images, pred_masks shape: (d, h, w) numpy array
        pred_masks must be class label not one hot label
        refine会自动把image和mask转移到refine model所在的gpu
    '''
    device = refiner.device
    refined_masks = np.zeros_like(pred_masks)
    for i in range(raw_images.shape[0]):
        start = time.time()
        image = np.stack((raw_images[i, :, :],) * 3, axis=-1)
        mask = pred_masks[i, :, :] * 255
        refined_mask = refiner.refine(image, mask, fast=False, L=1100)
        refined_masks[i, :, :] = refined_mask
        end = time.time()
        if show_progress:
            print(f"[INFO] XY slice {i+1} / {raw_images.shape[0]} refined on GPU {device}, cost {round(end-start,2)} s")
    # print(f"[INFO] refined masks: {np.unique(refined_masks)}")
    # foreground = refined_masks >= 128
    # background = refined_masks < 128
    # refined_masks[foreground] = 1
    # refined_masks[background] = 0
    refined_masks = (refined_masks >= 128).astype(np.uint8)
    return refined_masks

def refine_volume_xz(refiner, raw_images, pred_masks, show_progress=False):
    '''
        raw_images, pred_masks shape: (d, h, w) numpy array
        pred_masks must be class label not one hot label
        refine会自动把image和mask转移到refine model所在的gpu
    '''
    device = refiner.device
    refined_masks = np.zeros_like(pred_masks)
    for i in range(raw_images.shape[1]):
        start = time.time()
        image = np.stack((raw_images[:, i, :],) * 3, axis=-1)
        mask = pred_masks[:, i, :] * 255
        refined_mask = refiner.refine(image, mask, fast=False, L=1100)
        refined_masks[:, i, :] = refined_mask
        end = time.time()
        if show_progress:
            print(f"[INFO] XZ slice {i+1} / {raw_images.shape[1]} refined on GPU {device}, cost {round(end-start,2)} s")
    refined_masks = (refined_masks >= 128).astype(np.uint8)
    return refined_masks

def refine_volume_yz(refiner, raw_images, pred_masks, show_progress=False):
    '''
        raw_images, pred_masks shape: (d, h, w) numpy array
        pred_masks must be class label not one hot label
        refine会自动把image和mask转移到refine model所在的gpu
    '''
    device = refiner.device
    refined_masks = np.zeros_like(pred_masks)
    for i in range(raw_images.shape[2]):
        start = time.time()
        image = np.stack((raw_images[:, :, i],) * 3, axis=-1)
        mask = pred_masks[:, :, i] * 255
        refined_mask = refiner.refine(image, mask, fast=False, L=1100)
        refined_masks[:, :, i] = refined_mask
        end = time.time()
        if show_progress:
            print(f"[INFO] YZ slice {i+1} / {raw_images.shape[2]} refined on GPU {device}, cost {round(end-start,2)} s")
    refined_masks = (refined_masks >= 128).astype(np.uint8)
    return refined_masks

def sliding_window_infer(model, raw_images, patch_size, step, use_tta=False, deep_supervision=True, out_depth_index=0):
    '''
        Binary inference with sliding window
    '''
    print(f"[INFO] Start Sliding Window Inference")
    inference_start = time.time()
    if isinstance(raw_images, np.ndarray):
        infer_mask = np.zeros_like(raw_images, dtype=np.uint8)
    else:
        raise Exception(f'raw_images must be a Numpy array but got {type(raw_images)}')
    image_patches = patchify(raw_images, patch_size=patch_size, step=step)
    total_patches = image_patches.shape[0]*image_patches.shape[1]*image_patches.shape[2]
    infered_patch = 0
    z_s, y_s, x_s = patch_size # z_patch_size, y...
    if (raw_images.shape[0] - z_s) % step != 0:
        print(f"[INFO] (raw_images.shape[0] - z_s) % step = {(raw_images.shape[0] - z_s) % step}")
        print(f"[INFO] some z axis area may be not inferenced")
    if (raw_images.shape[1] - y_s) % step != 0:
        print(f"[INFO] (raw_images.shape[1] - y_s) % step = {(raw_images.shape[1] - y_s) % step}")
        print(f"[INFO] some y axis area may be not inferenced")
    if (raw_images.shape[2] - x_s) % step != 0:
        print(f"[INFO] (raw_images.shape[2] - y_s) % step = {(raw_images.shape[2] - x_s) % step}")
        print(f"[INFO] some x axis area may be not inferenced")
    model.eval()
    with torch.no_grad():
        for z in range(image_patches.shape[0]):
            for y in range(image_patches.shape[1]):
                for x in range(image_patches.shape[2]):
                    start = time.time()
                    image_patch = image_patches[z, y, x, :, :, :]
                    if use_tta:
                        hard_pred = infer_with_tta(model, torch.from_numpy(image_patch).unsqueeze(0)).argmax(0) #(d, h, w)
                    else:
                        hard_pred = infer_patch(model, torch.from_numpy(image_patch).unsqueeze(0), deep_supervision=deep_supervision, out_depth_index=out_depth_index).argmax(0)
                    infer_mask[z*step:z*step+z_s, y*step:y*step+y_s, x*step:x*step+x_s] = \
                        np.logical_or(infer_mask[z*step:z*step+z_s, y*step:y*step+y_s, x*step:x*step+x_s], hard_pred.cpu().numpy()).astype(np.uint8) # 0 and 1
                    end = time.time()
                    infered_patch += 1
                    print(f"[INFO] Inferenced {infered_patch} / {total_patches}, cost: {round(end-start, 2)} s")
    inference_end = time.time()
    print(f"[INFO] Cost of inference: {round(inference_end-inference_start, 2)} s")
    return infer_mask

def label_smoothing(labels, smoothing=0.1):
    """
    对 one-hot 标签进行平滑处理。
    
    Args:
        labels (torch.Tensor): 形状为 (B, num_classes, D, H, W) 的 one-hot 张量。
        smoothing (float): 平滑因子，范围在 [0, 1]。

    Returns:
        torch.Tensor: 平滑后的标签张量，形状与输入一致。
    """
    num_classes = labels.size(1)
    # 平滑后的值：每个类别分配的权重
    smooth_labels = (1.0 - smoothing) * labels + smoothing / num_classes
    return smooth_labels

def get_transforms():
    # Transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.10,
            ),
            ToTensord(keys=["image", "label"])
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=["image", "label"])
        ]
    )
    return train_transforms, val_transforms

def get_distance_map(mask):
    '''
        mask: (batch, num_cls, H, W, D)
        output: shape same as mask
    '''
    device = mask.device
    # Calculate distance map
    mask_np = mask.cpu().numpy()
    output = np.zeros_like(mask_np)
    batch_size = mask.shape[0]
    for batch_idx in range(batch_size):
        output[batch_idx] = distance_transform_edt(mask_np[batch_idx])
    # Normalize to [0,1]
    if output.max() > 0:
        output = output / output.max()
    return torch.from_numpy(output).to(device)

def get_signed_distance_map(mask):
    '''
        mask: (batch, 1, H, W, D) (binary mask: 0 for background, 1 for foreground)
        output: signed distance map, positive for foreground, negative for background
    '''
    device = mask.device
    mask_np = mask.cpu().numpy()
    output = np.zeros_like(mask_np, dtype=np.float32)
    batch_size = mask.shape[0]

    for batch_idx in range(batch_size):
        # Foreground distance
        fg_dist = distance_transform_edt(mask_np[batch_idx, 0])
        # Background distance
        bg_dist = distance_transform_edt(1 - mask_np[batch_idx, 0])
        fg_dist /= fg_dist.max() + 1e-6
        bg_dist /= bg_dist.max() + 1e-6
        # Signed distance map
        output[batch_idx, 0] = fg_dist - bg_dist

    # Normalize to [-1, 1]
    # max_dist = np.abs(output).max()
    # if max_dist > 0:
    #     output = output / max_dist

    return torch.from_numpy(output).to(device)


def get_contour(mask: torch.Tensor) -> torch.Tensor:
    '''
    mask: (batch, 1, H, W, D) (binary mask: 0 for background, 1 for foreground)
    output: binary contour mask, binary inner mask
    '''
    if mask.dim() != 5 or mask.shape[1] != 1:
        raise ValueError("Input mask must have shape (batch, 1, H, W, D).")
    
    device = mask.device
    mask_np = mask.cpu().numpy()
    kernel = np.ones((3, 3, 3), dtype=bool)  # 推荐使用更小的核，减少轮廓厚度

    contour = np.zeros_like(mask_np, dtype=np.float32)
    inner = np.zeros_like(mask_np, dtype=np.float32)
    batch_size = mask.shape[0]

    for batch_idx in range(batch_size):
        # 腐蚀操作
        erosion = ndimage.binary_erosion(mask_np[batch_idx, 0], structure=kernel)
        # 提取轮廓
        contour[batch_idx, 0] = mask_np[batch_idx, 0] & ~erosion
        inner[batch_idx, 0] = erosion
    
    return torch.from_numpy(contour).float().to(device), torch.from_numpy(inner).float().to(device)
