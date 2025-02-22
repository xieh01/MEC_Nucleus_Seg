import torch
import torch.nn as nn
import torch.nn.functional as F
from .pixel_shuffle3d import PixelShuffle3d, PixelUnshuffle3d
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from torch.cuda.amp import autocast
import torch.utils.checkpoint as cp

pixel_unshuffle = PixelUnshuffle3d(upscale_factor=2)
pixel_shuffle = PixelShuffle3d(upscale_factor=2)


def unfold3d(x, kernel_size, padding, stride):
    B, C, D, H, W = x.shape
    if isinstance(kernel_size, int):
        kD, kH, kW = kernel_size, kernel_size, kernel_size
    else:
        kD, kH, kW = kernel_size
    if isinstance(padding, int):
        pD, pH, pW = padding, padding, padding
    else:
        pD, pH, pW = padding
    if isinstance(stride, int):
        sD, sH, sW = stride, stride, stride
    else:
        sD, sH, sW = stride

    x = F.pad(x, (pW, pW, pH, pH, pD, pD))

    outD = (D + 2*pD - kD)//sD + 1
    outH = (H + 2*pH - kH)//sH + 1
    outW = (W + 2*pW - kW)//sW + 1

    x = x.unfold(2, kD, sD).unfold(3, kH, sH).unfold(4, kW, sW)
    x = x.permute(0, 1, 5, 6, 7, 2, 3, 4).contiguous()
    x = x.view(B, C, kD * kH * kW, outD, outH, outW)
    return x


def apply_filter_low_res(lr_feat, dy_filter, dy_kernel_size):
    B, C, D, H, W = lr_feat.shape
    _, K3, _, _, _ = dy_filter.shape
    assert dy_kernel_size ** 3 == K3

    alpfs = pixel_unshuffle(dy_filter)  # (B, K^3*8, D, H, W)
    alpfs = alpfs.view(B, 8, K3, D, H, W)  # (B, 8, K^3, D, H, W)
    alpfs = F.softmax(alpfs, dim=2)

    lr_unfold = unfold3d(lr_feat, kernel_size=dy_kernel_size, padding=1, stride=1)  # (B, C, K^3, D, H, W)

    filtered = []
    for i in range(8):
        filtered.append((lr_unfold * alpfs[:, i].unsqueeze(1)).sum(dim=2))  # (B, C, D, H, W)

    filtered = torch.stack(filtered, dim=1)  # (B, 8, C, D, H, W)
    filtered = filtered.permute(0, 2, 1, 3, 4, 5).contiguous().view(B, C * 8, D, H, W)  # (B, 8*C, D, H, W)
    output = pixel_shuffle(filtered)  # (B, C, 2*D, 2*H, 2*W)
    return output


def apply_filter_high_res(hr_feat, dy_filter, dy_kernel_size):
    _, K3, _, _, _ = dy_filter.shape
    assert dy_kernel_size ** 3 == K3

    dy_filter = F.softmax(dy_filter, dim=1) # (B, K^3, 2*D, 2*H, 2*W)

    hr_unfold = unfold3d(hr_feat, kernel_size=dy_kernel_size, padding=1, stride=1)  # (B, C, K^3, 2*D, 2*H, 2*W)

    filtered = (hr_unfold * dy_filter.unsqueeze(1)).sum(dim=2) # (B, C, 2*D, 2*H, 2*W)

    filtered = hr_feat + filtered # (B, C, 2*D, 2*H, 2*W) 

    return filtered


class FilteringBlock(nn.Module):
    def __init__(self, 
                 hr_channels,
                 lr_channels,
                 lowpass_kernel=3,
                 highpass_kernel=3,
                 feature_resample_groups=2,
                 use_checkpoint=True,
                 use_amp=True):
        super().__init__()
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.feature_resample_groups = feature_resample_groups
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp


        self.filter_low_mapping = nn.Conv3d(
            self.hr_channels,
            self.lowpass_kernel ** 3,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.filter_high_mapping = nn.Conv3d(
            self.hr_channels,
            self.highpass_kernel ** 3,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.init_up = UnetrUpBlock(spatial_dims=3, in_channels=self.lr_channels, out_channels=self.hr_channels, kernel_size=3, upsample_kernel_size=2, norm_name='instance')

        self._initialize_weights(initialization_type='he_leaky')


    def _initialize_weights(self, initialization_type='xavier', negative_slope=0.01):
        """
        Initialize network weights using Xavier or He initialization
        Args:
            initialization_type: str, 'xavier', 'he_relu', or 'he_leaky'
            negative_slope: float, slope for leaky relu (default 0.01)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if initialization_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                elif initialization_type == 'he_leaky':
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', 
                                        nonlinearity='leaky_relu', a=negative_slope)
                else:  # he_relu
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', 
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                if initialization_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)
                elif initialization_type == 'he_leaky':
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out',
                                        nonlinearity='leaky_relu', a=negative_slope)
                else:  # he_relu
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, hr_feat, lr_feat):
        '''
            hr: (B, hr_c, 2*D, 2*H, 2*W) high resolution feature map
            lr: (B, lr_c, D, H, W) low resolution feature map
        '''
        with autocast(enabled=self.use_amp):
            feat_fused = self.init_up(lr_feat, hr_feat) # (B, hr_channels, 2*D, 2*H, 2*W)

            filter_low = self.filter_low_mapping(feat_fused) # (B, K**3, 2*D, 2*H, 2*W)
            filter_high = self.filter_high_mapping(feat_fused) # (B, K**3, 2*D, 2*H, 2*W)

            if self.use_checkpoint:
                lr_refined = cp.checkpoint(apply_filter_low_res, lr_feat, filter_low, 3) # (B, lr_c, 2*D, 2*H, 2*W)
                hr_refined = cp.checkpoint(apply_filter_high_res, hr_feat, filter_high, 3) # (B, hr_c, 2*D, 2*H, 2*W)
            else:
                lr_refined = apply_filter_low_res(lr_feat, filter_low, 3) # (B, lr_c, 2*D, 2*H, 2*W)
                hr_refined = apply_filter_high_res(hr_feat, filter_high, 3) # (B, hr_c, 2*D, 2*H, 2*W)

            return lr_refined, hr_refined


class DyUpBlock(nn.Module):
    def __init__(
            self,
            in_channels, # also means lr_feat channels
            out_channels, # also means hr_feat channels
            norm_name='instance',
            kernel_size=3,
    ): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_name = norm_name
        self.up = FilteringBlock(hr_channels=out_channels, lr_channels=in_channels)
        self.out_conv = UnetResBlock(spatial_dims=3, in_channels=in_channels + out_channels, out_channels=out_channels, norm_name=norm_name, kernel_size=kernel_size, stride=1)

    def forward(self, lr_feat, hr_feat):
        lr_filtered_up, hr_filtered = self.up(lr_feat = lr_feat, hr_feat = hr_feat)
        out = torch.cat([lr_filtered_up, hr_filtered], dim=1) # (B, in_channels + out_channels, ...)
        out = self.out_conv(out)
        return out


