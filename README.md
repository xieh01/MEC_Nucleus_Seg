# SwinDDF
Dense Dynamic Fusion Network for 3D Segmentation of Complex-Shaped Nuclei

## Environment install
Clone this repository and install requirements.

```bash
git clone https://github.com/xieh01/MEC_Nucleus_Seg
cd MEC_Nucleus_Seg

conda create -n nuc_seg python=3.11
conda activate nuc_seg

pip install -r requirements.txt
```

## Data preparation
You can download our raw and patchfied data here https://drive.google.com/drive/folders/1hPkoBRguyCddqaq26-9SSGRqrm56BNAq?usp=drive_link

## Start training
```bash
torchrun --nproc_per_node=2 train_dist.py --dataset MEC-Nuclei --batch_size 1 --lr 0.0007 --dice 0.5 --ce 0.5 --epochs 500 --model SwinDFF --save_dir /output --save_ckpt
```
## Inference
```bash
python inference.py --images_dir data/block3/image --output_dir output --output_fname block3_seg.h5
```