# 1. 添加必要的导入
import os
import torch
import numpy as np
import argparse
import json
import time
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from monai.data import (
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.losses import DiceCELoss
from utils.loss import FocalFrequencyLoss3D
from monai.transforms import AsDiscrete
from monai.networks.nets import UNETR, UNet, SwinUNETR, BasicUNetPlusPlus
from nets.swin_dfn import SwinDFNet
from utils.utils import get_transforms
from utils.metrics import compute_accuracy, compute_mean_dice, compute_mean_iou, compute_mean_mcc


# 2. 添加分布式训练初始化函数
def init_distributed():
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    # 设置当前设备
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# 3. 修改数据加载器创建部分
def create_data_loaders(train_ds, val_ds, test_ds, batch_size, num_workers=2):
    train_sampler = DistributedSampler(train_ds)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_sampler

def create_model(model_name, img_size=(64,64,64), device=None):
    if model_name == 'UNet':
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
            img_size=img_size,
        ).to(device)
    elif model_name == 'SwinUNETR':
        model = SwinUNETR(img_size=img_size, in_channels=1, out_channels=2, use_v2=True).to(device)
    elif model_name == 'SwinDFNet':
        model = SwinDFNet(img_size=img_size, in_channels=1, out_channels=2).to(device)
    elif model_name == 'UNetPP':
        model = BasicUNetPlusPlus(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            deep_supervision=True
        ).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    return model

def get_dataset(dataset_name):
    if dataset_name == 'NucMM-M':
        json_path = os.path.normpath("./json/NucMM_M.json")
        img_size = (64, 64, 64)
        ScaleIntensityRangedKeys = ["image"] # the range of NucMM label value is 0 and 1, no need to scale
    elif dataset_name == 'Nucv4':
        json_path = os.path.normpath("./json/Nuc_v4.json")
        img_size = (128, 128, 128)

    data_dir = os.path.normpath("../")
    logdir = os.path.normpath("../output/")

    if os.path.exists(logdir) is False:
        os.mkdir(logdir)

    train_transforms, val_transforms = get_transforms()
    with open(json_path, "r") as json_f:
        # load binary file to dict
        json_data = json.load(json_f)

    train_data = json_data['training']
    val_data = json_data['validation']
    test_data = json_data['test']

    tr_num = json_data['numTraining']
    val_num = json_data['numValidation']
    test_num = json_data['numTest']

    print(f"[INFO] Train data num:{tr_num}, Val data num:{val_num}, Test data num:{test_num}")

    # merge dir
    for idx, _each_d in enumerate(train_data):
        # if '\\' in os.path.join(data_dir, train_data[idx]['image']), replace it with '/'
        if '\\' in os.path.join(data_dir, train_data[idx]['image']):
            train_data[idx]['image'] = os.path.join(data_dir, train_data[idx]['image']).replace('\\', '/')
        else:
            train_data[idx]['image'] = os.path.join(data_dir, train_data[idx]['image'])
        if '\\' in os.path.join(data_dir, train_data[idx]['label']):
            train_data[idx]['label'] = os.path.join(data_dir, train_data[idx]['label']).replace('\\', '/')
        else:
            train_data[idx]['label'] = os.path.join(data_dir, train_data[idx]['label'])

    for idx, _each_d in enumerate(val_data):
        if '\\' in os.path.join(data_dir, val_data[idx]['image']):
            val_data[idx]['image'] = os.path.join(data_dir, val_data[idx]['image']).replace('\\', '/')
        else:
            val_data[idx]['image'] = os.path.join(data_dir, val_data[idx]['image'])
        if '\\' in os.path.join(data_dir, val_data[idx]['label']):
            val_data[idx]['label'] = os.path.join(data_dir, val_data[idx]['label']).replace('\\', '/')
        else:
            val_data[idx]['label'] = os.path.join(data_dir, val_data[idx]['label'])
    for idx, _each_d in enumerate(test_data):
        test_data[idx]['image'] = os.path.join(data_dir, test_data[idx]['image'])
        test_data[idx]['label'] = os.path.join(data_dir, test_data[idx]['label'])

    # define dataloader
    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    test_ds = Dataset(data=test_data, transform=val_transforms)

    return train_ds, val_ds, test_ds, img_size, tr_num, val_num, test_num


# 4. 主训练代码修改
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="param of training")
    parser.add_argument('--dataset', type=str, help='dataset name', default='NucMM-Z')
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--dice', type=float, help='lambda dice', default=1.0)
    parser.add_argument('--ce', type=float, help='lambda ce', default=1.0)
    parser.add_argument('--epochs', type=int, help='max epoch', default=200)
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--save_dir', type=str, help='ckpt save dir', default='/output')
    parser.add_argument('--save_ckpt', help='save ckpt', action='store_true')
    parser.add_argument('--weight_setting', type=int, help='deep supervision calc weight method', default=1)
    parser.add_argument('--ns', type=int, help='net setting', default=12)
    args = parser.parse_args()

    # 读取参数
    dataset = args.dataset
    batch_size = args.batch_size
    lr = args.lr
    lambda_dice = args.dice
    lambda_ce = args.ce
    max_epochs = args.epochs
    model_name = args.model
    save_dir = args.save_dir
    save_ckpt = args.save_ckpt
    weight_setting = args.weight_setting
    
    eval_num = 1

    # using early stopping
    patience = 100
    counter = 0
    best_val_dice = 0.0

    # 初始化分布式训练
    local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 创建数据集
    train_ds, val_ds, test_ds, img_size, tr_num, val_num, test_num = get_dataset(dataset)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, train_sampler = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size
    )
    
    # 创建模型并移到对应设备
    model = create_model(model_name, img_size=img_size, ns=args.ns, device=device) # 你的模型创建函数
    # model = model.to(device)
    # 包装为DDP模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=lambda_dice, lambda_ce=lambda_ce)
    freq_loss = FocalFrequencyLoss3D(alpha=1.0)
    freq_weight = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    step = 0
    tr_step_losses = []
    val_step_losses = []
    tr_losses_per_epoch = []
    val_losses_per_epoch = []
    tr_dice_per_epoch = []
    val_dice_per_epoch = []
    tr_iou_per_epoch = []
    val_iou_per_epoch = []
    tr_acc_per_epoch = []
    val_acc_per_epoch = []
    tr_mcc_per_epoch = []
    val_mcc_per_epoch = []

    # 训练循环
    for epoch in range(max_epochs):
        # 设置采样器的epoch
        train_sampler.set_epoch(epoch)
        model.train()
        training_dice_per_sample = []
        training_iou_per_sample = []
        training_acc_per_sample = []
        training_mcc_per_sample = []
        training_loss = 0
        start_time = time.time()
        # 训练代码
        for batch_data in train_loader:
            step += 1
            x, y = (
                batch_data['image'].to(device),
                batch_data['label'].to(device)
            )
            # ... 其他训练代码 ...
            optimizer.zero_grad()
            label_list = decollate_batch(y)
            label_convert = [post_label(label_tensor) for label_tensor in label_list]
            if model_name == 'FFSwinUNETR':
                logit_map, map1_1, map1_2, map1_3, map1_4 = model(x)
                # 深度递减 + 自适应权重策略
                with torch.no_grad():
                    # 计算每个输出的 Dice 分数（防止除0）
                    dice_scores = []
                    iou_scores = []
                    for pred in [map1_1, map1_2, map1_3, map1_4, logit_map]:
                        pred_list = decollate_batch(pred)
                        pred_convert = [post_pred(p) for p in pred_list]
                        dice = sum(compute_mean_dice(pred_convert[i], label_convert[i]) for i in range(len(pred_convert))) / len(pred_convert)
                        dice_scores.append(dice + 1e-6)
                        iou = sum(compute_mean_iou(pred_convert[i], label_convert[i]) for i in range(len(pred_convert))) / len(pred_convert)
                        iou_scores.append(iou + 1e-6)

                    # 深度递减权重（层越深权重越高）
                    depth_weights = [0.1, 0.2, 0.3, 0.4, 0.5]  # 根据层次调整

                    # 自适应权重（表现差的层权重高）
                    dice_weights = [(1 - dice) for dice in dice_scores]
                    iou_weights = [(1 - iou) for iou in iou_scores]
                    # total_weight = sum([d * a for d, a in zip(depth_weights, dice_weights)])
                    if weight_setting == 1:
                        total_weight = sum([dep_w * dice_w * iou_w for dep_w, dice_w, iou_w in zip(depth_weights, dice_weights, iou_weights)])
                        combined_weights = [(d * a * i) / total_weight for d, a, i in zip(depth_weights, dice_weights, iou_weights)]
                    elif weight_setting == 2:
                        total_weight = sum([d * a for d, a in zip(depth_weights, dice_weights)])
                        combined_weights = [(d * a) / total_weight for d, a in zip(depth_weights, dice_weights)]
                    elif weight_setting == 3:
                        total_weight = sum([d * i for d, i in zip(depth_weights, iou_weights)])
                        combined_weights = [(d * i) / total_weight for d, i in zip(depth_weights, iou_weights)]
                    elif weight_setting == 4:
                        total_weight = sum([a * i for a, i in zip(dice_weights, iou_weights)])
                        combined_weights = [(a * i) / total_weight for a, i in zip(dice_weights, iou_weights)]
                    elif weight_setting == 5:
                        # mean weight
                        combined_weights = [(1 / len(dice_weights)) for _ in range(len(dice_weights))]
                    elif weight_setting == 6:
                        # only depth weight(norm)
                        total_weight = sum(depth_weights)
                        combined_weights = [(d / total_weight) for d in depth_weights]
                    elif weight_setting == 7:
                        # only dice weight(norm)
                        total_weight = sum(dice_weights)
                        combined_weights = [(d / total_weight) for d in dice_weights]
                    elif weight_setting == 8:
                        # only iou weight(norm)
                        total_weight = sum(iou_weights)
                        combined_weights = [(i / total_weight) for i in iou_weights]
                    elif weight_setting == 9:
                        # no deep supervision
                        combined_weights = [0.0, 0.0, 0.0, 0.0, 1.0]
                    elif weight_setting == 10:
                        combined_weights = [0.05, 0.1, 0.15, 0.2, 1.0]
                    elif weight_setting == 11:
                        combined_weights = [0.1, 0.2, 0.3, 0.4, 1.0]
                    elif weight_setting == 12:
                        dice_iou_product = [(1 - dice) * (1 - iou) for dice, iou in zip(dice_scores, iou_scores)][:-1] # exclude the last one
                        total_weight = sum(dice_iou_product)
                        combined_weights = [i / total_weight for i in dice_iou_product]
                        combined_weights.append(1.0) # add the last one
                        print(f"[INFO] combined_weights:{combined_weights}")
                    elif weight_setting == 13:
                        dice_iou_product = [(1 - dice) * (1 - iou) for dice, iou in zip(dice_scores, iou_scores)][:-1] # exclude the last one
                        total_weight = sum(dice_iou_product)
                        combined_weights = [i / total_weight for i in dice_iou_product]
                        depth_weights = [0.01, 0.05, 0.1, 0.2]
                        # final weights equal to the product of combined_weights and depth_weights
                        combined_weights = [d * c for d, c in zip(depth_weights, combined_weights)]
                        combined_weights.append(1.0)
                        print(f"[INFO] combined_weights:{combined_weights}")
                    elif weight_setting == 14:
                        combined_weights = [0.15, 0.25, 0.5, 0.75, 1.0]
                    elif weight_setting == 15:
                        combined_weights = [0.0, 0.0, 0.0, 0.75, 1.0]
                    elif weight_setting == 16:
                        combined_weights = [0.0, 0.0, 0.5, 0.75, 1.0]


                # 加权计算分割损失
                tr_seg_loss = sum(w * loss_function(pred, y) for w, pred in zip(combined_weights, [map1_1, map1_2, map1_3, map1_4, logit_map]))
            elif model_name == 'UNetPP':
                logit_map = model(x)[-1]
                tr_seg_loss = loss_function(logit_map, y)
            else:
                logit_map = model(x)
                tr_seg_loss = loss_function(logit_map, y)
            tr_freq_loss = freq_weight * freq_loss(torch.softmax(logit_map, dim=1), y)
            total_training_loss = tr_seg_loss + tr_freq_loss
            total_training_loss.backward()
            optimizer.step()
            # calc and record loss and metric
            tr_step_losses.append(total_training_loss.item())
            training_loss += total_training_loss.item()
            if local_rank == 0:
                print(f"step : {step} : training loss: {total_training_loss.item()}, seg loss: {tr_seg_loss.item()}, freq loss: {tr_freq_loss.item()}")
            with torch.no_grad():
                logit_map_list = decollate_batch(logit_map)
                output_convert = [post_pred(logit_map_tensor) for logit_map_tensor in logit_map_list]
                # calc dice score
                for i in range(len(output_convert)):
                    training_dice_per_sample.append(compute_mean_dice(output_convert[i], label_convert[i]))
                    training_iou_per_sample.append(compute_mean_iou(output_convert[i], label_convert[i]))
                    training_mcc_per_sample.append(compute_mean_mcc(output_convert[i], label_convert[i]))
                    training_acc_per_sample.append(compute_accuracy(output_convert[i], label_convert[i]))
        print(f"Num of training samples: {len(training_dice_per_sample)}")
        training_dice = np.mean(training_dice_per_sample)
        training_iou = np.mean(training_iou_per_sample)
        training_mcc = np.mean(training_mcc_per_sample)
        training_acc = np.mean(training_acc_per_sample)
        training_loss = training_loss / len(train_loader)
        tr_dice_per_epoch.append(training_dice)
        tr_iou_per_epoch.append(training_iou)
        tr_mcc_per_epoch.append(training_mcc)
        tr_acc_per_epoch.append(training_acc)
        tr_losses_per_epoch.append(training_loss)
        end_time = time.time()
        if local_rank == 0:
            print(f"Train epoch {epoch+1} / {max_epochs}: dice {round(training_dice, 4)}, iou {round(training_iou, 4)}, acc {round(training_acc, 4)}, mcc {round(training_mcc, 4)}")
            print(f"Train loss: {round(training_loss, 5)}, cost {round(end_time-start_time, 2)}s")
            
        # 验证代码
        if (epoch + 1) % eval_num == 0:
            model.eval()
            # ... 验证代码 ...
            start_time = time.time()
            model.eval()
            validating_dice_per_sample = []
            validating_iou_per_sample = []
            validating_mcc_per_sample = []
            validating_acc_per_sample = []
            validating_loss = 0
            if local_rank == 0:
                print("start validation...")
            with torch.no_grad():
                for val_batch in val_loader:
                    x, y = (
                        val_batch['image'].to(local_rank),
                        val_batch['label'].to(local_rank)
                    )
                    label_list = decollate_batch(y)
                    label_convert = [post_label(label_tensor) for label_tensor in label_list]

                    if model_name == 'FFSwinUNETR':
                        logit_map, map1_1, map1_2, map1_3, map1_4 = model(x)

                        # 计算每个输出的 Dice 分数（防止除0）
                        dice_scores = []
                        iou_scores = []
                        for pred in [map1_1, map1_2, map1_3, map1_4, logit_map]:
                            pred_list = decollate_batch(pred)
                            pred_convert = [post_pred(p) for p in pred_list]
                            dice = sum(compute_mean_dice(pred_convert[i], label_convert[i]) for i in range(len(pred_convert))) / len(pred_convert)
                            dice_scores.append(dice + 1e-6)
                            iou = sum(compute_mean_iou(pred_convert[i], label_convert[i]) for i in range(len(pred_convert))) / len(pred_convert)
                            iou_scores.append(iou + 1e-6)

                        # 深度递减权重（层越深权重越高）
                        depth_weights = [0.1, 0.2, 0.3, 0.4, 0.5]  # 根据层次调整

                        # 自适应权重（表现差的层权重高）
                        dice_weights = [(1 - dice) for dice in dice_scores]
                        iou_weights = [(1 - iou) for iou in iou_scores]
                        if weight_setting == 1:
                            total_weight = sum([dep_w * dice_w * iou_w for dep_w, dice_w, iou_w in zip(depth_weights, dice_weights, iou_weights)])
                            combined_weights = [(d * a * i) / total_weight for d, a, i in zip(depth_weights, dice_weights, iou_weights)]
                        elif weight_setting == 2:
                            total_weight = sum([d * a for d, a in zip(depth_weights, dice_weights)])
                            combined_weights = [(d * a) / total_weight for d, a in zip(depth_weights, dice_weights)]
                        elif weight_setting == 3:
                            total_weight = sum([d * i for d, i in zip(depth_weights, iou_weights)])
                            combined_weights = [(d * i) / total_weight for d, i in zip(depth_weights, iou_weights)]
                        elif weight_setting == 4:
                            total_weight = sum([a * i for a, i in zip(dice_weights, iou_weights)])
                            combined_weights = [(a * i) / total_weight for a, i in zip(dice_weights, iou_weights)]
                        elif weight_setting == 5:
                            # mean weight
                            combined_weights = [(1 / len(dice_weights)) for _ in range(len(dice_weights))]
                        elif weight_setting == 6:
                            # only depth weight(norm)
                            total_weight = sum(depth_weights)
                            combined_weights = [(d / total_weight) for d in depth_weights]
                        elif weight_setting == 7:
                            # only dice weight(norm)
                            total_weight = sum(dice_weights)
                            combined_weights = [(d / total_weight) for d in dice_weights]
                        elif weight_setting == 8:
                            # only iou weight(norm)
                            total_weight = sum(iou_weights)
                            combined_weights = [(i / total_weight) for i in iou_weights]
                        elif weight_setting == 9:
                            # no deep supervision
                            combined_weights = [0.0, 0.0, 0.0, 0.0, 1.0]
                        elif weight_setting == 10:
                            combined_weights = [0.05, 0.1, 0.15, 0.2, 1.0]
                        elif weight_setting == 11:
                            combined_weights = [0.1, 0.2, 0.3, 0.4, 1.0]
                        elif weight_setting == 12:
                            dice_iou_product = [(1 - dice) * (1 - iou) for dice, iou in zip(dice_scores, iou_scores)][:-1] # exclude the last one
                            total_weight = sum(dice_iou_product)
                            combined_weights = [i / total_weight for i in dice_iou_product]
                            combined_weights.append(1.0) # add the last one
                            print(f"[INFO] combined_weights:{combined_weights}")
                        elif weight_setting == 13:
                            dice_iou_product = [(1 - dice) * (1 - iou) for dice, iou in zip(dice_scores, iou_scores)][:-1] # exclude the last one
                            total_weight = sum(dice_iou_product)
                            combined_weights = [i / total_weight for i in dice_iou_product]
                            depth_weights = [0.01, 0.05, 0.1, 0.2]
                            # final weights equal to the product of combined_weights and depth_weights
                            combined_weights = [d * c for d, c in zip(depth_weights, combined_weights)]
                            combined_weights.append(1.0)
                            print(f"[INFO] combined_weights:{combined_weights}")
                        elif weight_setting == 14:
                            combined_weights = [0.15, 0.25, 0.5, 0.75, 1.0]
                        elif weight_setting == 15:
                            combined_weights = [0.0, 0.0, 0.0, 0.75, 1.0]
                        elif weight_setting == 16:
                            combined_weights = [0.0, 0.0, 0.5, 0.75, 1.0]

                        # 加权计算分割损失
                        val_seg_loss = sum(w * loss_function(pred, y) for w, pred in zip(combined_weights, [map1_1, map1_2, map1_3, map1_4, logit_map]))
                    elif model_name == 'UNetPP':
                        logit_map = model(x)[-1]
                        val_seg_loss = loss_function(logit_map, y)
                    else:
                        logit_map = model(x)
                        val_seg_loss = loss_function(logit_map, y)
                    val_freq_loss = freq_weight * freq_loss(torch.softmax(logit_map, dim=1), y)
                    total_val_loss = val_seg_loss + val_freq_loss
                    val_step_losses.append(total_val_loss.item())
                    validating_loss += total_val_loss.item()
                    if local_rank == 0:
                        print(f"val loss: {total_val_loss.item()}, seg loss: {val_seg_loss.item()}, freq loss: {val_freq_loss.item()}")
                    logit_map_list = decollate_batch(logit_map)
                    output_convert = [post_pred(logit_map_tensor) for logit_map_tensor in logit_map_list]
                    label_list = decollate_batch(y)
                    label_convert = [post_label(label_tensor) for label_tensor in label_list]
                    # calc metrics score
                    for i in range(len(output_convert)):
                        validating_dice_per_sample.append(compute_mean_dice(output_convert[i], label_convert[i]))
                        validating_iou_per_sample.append(compute_mean_iou(output_convert[i], label_convert[i]))
                        validating_mcc_per_sample.append(compute_mean_mcc(output_convert[i], label_convert[i]))
                        validating_acc_per_sample.append(compute_accuracy(output_convert[i], label_convert[i]))
                print(f"Num of validating samples: {len(validating_dice_per_sample)}")
                validating_dice = np.mean(validating_dice_per_sample)
                validating_iou = np.mean(validating_iou_per_sample)
                validating_mcc = np.mean(validating_mcc_per_sample)
                validating_acc = np.mean(validating_acc_per_sample)
                validating_loss = validating_loss / len(val_loader)
                val_dice_per_epoch.append(validating_dice)
                val_iou_per_epoch.append(validating_iou)
                val_mcc_per_epoch.append(validating_mcc)
                val_acc_per_epoch.append(validating_acc)
                val_losses_per_epoch.append(validating_loss)
                end_time = time.time()
                if local_rank == 0:
                    print(f"Val epoch {epoch+1} / {max_epochs}: dice {round(validating_dice, 4)}, iou {round(validating_iou, 4)}, acc {round(validating_acc, 4)}, mcc {round(validating_mcc, 4)}")
                    print(f"Val loss: {round(validating_loss, 5)}, cost {round(end_time-start_time, 2)}s")
            if validating_dice >= best_val_dice:
                counter = 0
                best_val_dice = validating_dice
                if save_ckpt:
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_dice_weights.pth"))
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 只在主进程上保存模型和输出日志
        if local_rank == 0:
            if save_ckpt and validating_dice >= np.max(val_dice_per_epoch):
                torch.save(model.module.state_dict(), 
                         os.path.join(save_dir, "best_dice_weights.pth"))
            # 画图和打印日志
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))

            # 第一个子图
            axs[0, 0].plot(tr_losses_per_epoch, label='Train Loss')
            axs[0, 0].plot(val_losses_per_epoch, label='Validation Loss')
            axs[0, 0].set_title('Epoch Losses')
            axs[0, 0].legend()

            # 第二个子图
            axs[0, 1].plot(tr_dice_per_epoch, label='Train Dice')
            axs[0, 1].plot(val_dice_per_epoch, label='Validation Dice')
            axs[0, 1].set_title('Dice Coefficients')
            axs[0, 1].legend()

            # 第三个子图
            axs[1, 0].plot(tr_iou_per_epoch, label='Train IoU')
            axs[1, 0].plot(val_iou_per_epoch, label='Validation IoU')
            axs[1, 0].set_title('IoU')
            axs[1, 0].legend()

            # 第四个子图
            axs[1, 1].plot(tr_acc_per_epoch, label='Train Accuracy')
            axs[1, 1].plot(val_acc_per_epoch, label='Validation Accuracy')
            axs[1, 1].set_title('Accuracy')
            axs[1, 1].legend()

            # 调整布局
            plt.tight_layout()

            # 保存图像
            plt.savefig('../output/output.png')

    if local_rank == 0:
        print(f"Config :{args}")
        print("Training Finished")
        print(f"Best val dice: {round(np.max(val_dice_per_epoch), 4)}, ep {np.argmax(val_dice_per_epoch) + 1}")
        print(f"Best val iou: {round(np.max(val_iou_per_epoch), 4)}, ep {np.argmax(val_iou_per_epoch) + 1}")
        print(f"Best val mcc: {round(np.max(val_mcc_per_epoch), 4)}, ep {np.argmax(val_mcc_per_epoch) + 1}")
        print(f"Best val accuracy: {round(np.max(val_acc_per_epoch), 4)}, ep {np.argmax(val_acc_per_epoch) + 1}")
        
        if dataset in ['Nucv4', 'Lucchiv3']:
            # test
            print("start testing...")
            # 加载模型权重
            checkpoint = torch.load(os.path.join(save_dir, "best_dice_weights.pth"))
            # 模型是用 DDP 包装的，添加 'module.' 前缀
            new_state_dict = {}
            for k, v in checkpoint.items():
                if not k.startswith('module.'):
                    new_state_dict[f'module.{k}'] = v
                else:
                    new_state_dict[k] = v
            # 加载新的状态字典
            model.load_state_dict(new_state_dict)
            testing_dice = 0
            testing_iou = 0
            testing_mcc = 0
            testing_acc = 0
            for test_batch in test_loader:
                x, y = (
                    test_batch['image'].to(device),
                    test_batch['label'].to(device)
                )
                with torch.no_grad():
                    if model_name == 'FFSwinUNETR':
                        logit_map, map1_1, map1_2, map1_3, map1_4 = model(x)
                    else:
                        logit_map = model(x)
                    logit_map_list = decollate_batch(logit_map)
                    output_convert = [post_pred(logit_map_tensor) for logit_map_tensor in logit_map_list]
                    label_list = decollate_batch(y)
                    label_convert = [post_label(label_tensor) for label_tensor in label_list]
                    for i in range(len(output_convert)):
                        testing_dice += compute_mean_dice(output_convert[i], label_convert[i])
                        testing_iou += compute_mean_iou(output_convert[i], label_convert[i])
                        testing_mcc += compute_mean_mcc(output_convert[i], label_convert[i])
                        testing_acc += compute_accuracy(output_convert[i], label_convert[i])
            testing_dice /= test_num
            testing_iou /= test_num
            testing_mcc /= test_num
            testing_acc /= test_num
            print(f"Test dice {round(testing_dice, 4)}, iou {round(testing_iou, 4)}, acc {round(testing_acc, 4)}, mcc {round(testing_mcc, 4)}")


    # 清理进程组
    dist.destroy_process_group()
