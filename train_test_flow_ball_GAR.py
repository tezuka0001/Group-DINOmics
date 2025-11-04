import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import copy
import time
import random
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, multilabel_confusion_matrix
from sklearn.neighbors import NearestNeighbors
import wandb
import matplotlib.pyplot as plt
import pandas as pd

import models.models_flow_ball as models
from util.utils import *
from dataloader.dataloader_bbox_flow_detector import read_dataset
from sklearn.manifold import TSNE
from tqdm import tqdm

# import lora
from omegaconf import OmegaConf
import yaml
from lama.bin.predict_inpaint import *
import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

from PIL import Image

parser = argparse.ArgumentParser(description='Ball Detect Group Activity Recognition')

# Dataset specification
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')
parser.add_argument('--data_path', default='./Dataset/', type=str, help='data path')
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', default='seg_samp', help='random sampling strategy, if you want to use full frames, please set full_frames')
parser.add_argument('--num_frame', default=18, type=int, help='number of frames for each clip')
parser.add_argument('--num_total_frame', default=72, type=int, help='number of total frames for each clip')
parser.add_argument('--num_activities', default=9, type=int, help='number of activity classes in volleyball dataset')

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=30, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=2, type=int, help='print frequency')
parser.add_argument('--batch', default=4, type=int, help='Batch size')
parser.add_argument('--feature_batch', default=2, type=int, help='Batch size')
parser.add_argument('--test_batch', default=2, type=int, help='Test batch size')
parser.add_argument('--lr', default=1e-6, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=1e-5, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')
parser.add_argument('--nheads', default=4, type=int, help='number of heads for partial context aggregation')

# GPU
parser.add_argument('--device', default="0", type=str, help='GPU device')

parser.add_argument('--hidden_size', default=1024, type=int, help='hidden size')
parser.add_argument('--detector', action='store_true', help='use detector')

# backbone parameters
parser.add_argument('--backbone_learnable', action='store_true', help='use learnable last layer')
parser.add_argument('--backbone_full_learnable', action='store_true', help='use full learnable last layer')
parser.add_argument('--backbone_learnable_layers', default=1, type=int, help='number of learnable layers')
parser.add_argument('--backbone', default='dinov3', type=str, help='backbone model, dinov2, clip, ViT, MAE, resnet50, vgg16, vgg19')
parser.add_argument('--ViT_arch', default='vit-l', type=str, help='vit-l, vit-b')
parser.add_argument('--use_lora', action='store_true', help='use LoRA for backbone')
parser.add_argument('--linear_probing', action='store_true', help='use linear probing for backbone')

parser.add_argument('--ball_mask', action='store_true', help='use ball mask')
parser.add_argument('--random_mask', action='store_true', help='use random mask')
parser.add_argument('--ball_inpaint', action='store_true', help='use ball inpaint')
parser.add_argument('--people_mask', action='store_true', help='use people mask')
parser.add_argument('--ball_pred', action='store_true', help='use ball prediction')
parser.add_argument('--flow_pred', action='store_true', help='use flow prediction')

parser.add_argument('--supervised', action='store_true', help='use supervised loss')
parser.add_argument('--pretrained_weights', type=str, default='', help='pretrained weights path')
parser.add_argument('--fix_model', action='store_true', help='fix model parameters')

parser.add_argument('--temporal_mask', action='store_true', help='use temporal mask')
parser.add_argument('--future_mask', action='store_true', help='use feature mask')
parser.add_argument('--test_time_mask', action='store_true', help='use test time mask')
parser.add_argument('--spatial_loss', action='store_true', help='use spatial loss')
parser.add_argument('--temporal_loss', action='store_true', help='use temporal loss')
parser.add_argument('--spatial_flow_loss', action='store_true', help='use spatial loss')
parser.add_argument('--temporal_flow_loss', action='store_true', help='use temporal loss')
parser.add_argument('--mlp_comp', action='store_true', help='use mlp compression')
parser.add_argument('--trans_comp', action='store_true', help='use transformer compression')
parser.add_argument('--loc_guide', default='spatial_temporal_loc', type=str, help='localization guidance type: none, spatial_loc, temporal_loc, spatial_temporal_loc')
parser.add_argument('--comp_dim', default=128, type=int, help='compression dimension')
parser.add_argument('--w_flow', default=1.0, type=float, help='weight of flow loss')
parser.add_argument('--w_ball', default=1.0, type=float, help='weight of ball loss')
parser.add_argument('--w_sup', default=1.0, type=float, help='weight of supervised loss')

# lama param
parser.add_argument('--ball_lama', action='store_true', help='use ball inpaint with LaMa')
parser.add_argument('--frame_random', action='store_true', help='use frame random')
parser.add_argument('--batch_random', action='store_true', help='use batch random')
parser.add_argument('--inpaint_prob', type=float, default=1.0, help='probability of inpainting')
parser.add_argument('--people_lama', action='store_true', help='use people inpaint with LaMa')
parser.add_argument('--inpaint_people_prob', type=float, default=0.5, help='probability of people inpainting')
parser.add_argument('--people_mask_scale', type=float, default=1.0, help='scale for people mask')
parser.add_argument('--lama_model_dir', default='./lama/big-lama/', type=str, help='LaMa model directory')
parser.add_argument('--checkpoint_name', default='best.ckpt', type=str, help='LaMa checkpoint name')
parser.add_argument('--mask_shape', default='circle', type=str, help='mask shape: circle or square')

# detector-freeを使う場合の引数
parser.add_argument('--ViT_Blocks', default=0, type=int, help='number of blocks for ViT backbone')
parser.add_argument('--detector_free', action='store_true', help='use detector free backbone')
parser.add_argument('--num_tokens', default=12, type=int, help='number of tokens for backbone')
parser.add_argument('--ffn_dim', default=2048, type=int, help='feed forward network dimension')
parser.add_argument('--enc_layers', default=6, type=int, help='number of encoder layers')
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')

args = parser.parse_args()
args.use_flow = False
args.use_flow_numpy = args.flow_pred

best_hit_rate = 0.0
best_hit_epoch = 0
best_f1 = 0.0
best_f1_epoch = 0
best_model_path = None

def save_best_model(epoch, model, optimizer, scheduler, accuracy, save_path):
    global best_model_path
    result_path = os.path.join(save_path, 'epoch%d_%.2f.pth' % (epoch, accuracy))
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'accuracy': accuracy
    }
    torch.save(state, result_path)
    if best_model_path is not None and os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Removed old model: {best_model_path}")
    best_model_path = result_path
    print(f"Saved new best model at epoch {epoch} with accuracy {accuracy:.2f}% to {result_path}")

@torch.no_grad()
def test_evaluation(train_loader_for_val, test_loader, model, args, save_path):
    model.eval()

    train_features_list = []
    train_labels_list   = []
    for sample in tqdm(train_loader_for_val, desc="Extracting Train Features for Test"):
        if args.use_flow or args.use_flow_numpy:
            images, activities, *_ = sample
        else:
            images, activities, *_ = sample
        images   = images.cuda()
        bboxes   = sample[-2].cuda()
        ret_dic  = model({'images': images, 'bboxes': bboxes})
        train_features_list.append(ret_dic['video_features'].cpu().numpy())
        train_labels_list.append(activities[:, 0].cpu().numpy())
    train_features = np.vstack(train_features_list)
    train_labels   = np.hstack(train_labels_list)

    test_features_list = []
    test_labels_list   = []
    cls_preds_list     = []
    correct_cls = 0
    total_cls   = 0

    for sample in tqdm(test_loader, desc="Extracting Test Features"):
        if args.use_flow or args.use_flow_numpy:
            images, activities, *_ = sample
        else:
            images, activities, *_ = sample

        images   = images.cuda()
        labels   = activities[:, 0].cuda()    # (B,)
        bboxes   = sample[-2].cuda()
        ret_dic  = model({'images': images, 'bboxes': bboxes})

        test_features_list.append(ret_dic['video_features'].cpu().numpy())
        test_labels_list.append(labels.cpu().numpy())

        activities_score = ret_dic['activities_score']
        preds  = activities_score.argmax(dim=-1)         # (B,)
        correct_cls += (preds == labels).sum().item()
        total_cls   += labels.size(0)
        cls_preds_list.append(preds.cpu().numpy())

    test_features = np.vstack(test_features_list)
    test_labels   = np.hstack(test_labels_list)
    accuracy      = correct_cls / total_cls * 100.0
    cls_preds     = np.hstack(cls_preds_list)

    k_list     = [1, 2, 3, 4, 5]
    nbrs_k     = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(train_features)
    _, indices_k = nbrs_k.kneighbors(test_features)

    hit_counts    = {k: 0 for k in k_list}
    precision_sum = {k: 0.0 for k in k_list}
    for i, neighbors in enumerate(indices_k):
        neigh_lbls = train_labels[neighbors]
        for k in k_list:
            correct_k = np.sum(neigh_lbls[:k] == test_labels[i])
            precision_sum[k] += correct_k / k
            if correct_k > 0:
                hit_counts[k] += 1

    total_q    = len(test_labels)
    hit_rates  = {k: hit_counts[k] / total_q * 100.0 for k in k_list}
    precisions = {k: precision_sum[k] / total_q * 100.0 for k in k_list}

    cm = confusion_matrix(test_labels, cls_preds)
    cm_df = pd.DataFrame(cm)
    csv_path = os.path.join(save_path, "confusion_matrix.csv")
    cm_df.to_csv(csv_path, index=False)
    plot_confusion_matrices_from_dir(args, save_path)
    jpg_path = os.path.join(save_path, "confusion_matrix.jpg")

    per_class_acc = cm.diagonal().astype(float) / cm.sum(axis=1)
    mean_per_class_acc = np.nanmean(per_class_acc) * 100.0

    for features, labels, tag in [
        (train_features, train_labels, "train"),
        (test_features,  test_labels,  "test")
    ]:
        tsne = TSNE(n_components=2, random_state=42)
        z = tsne.fit_transform(features)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', s=20)
        plt.colorbar(scatter)
        plt.title(f"t-SNE of {tag} features")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        out_path = os.path.join(save_path, f"feature_space_{tag}_tsne.jpg")
        plt.savefig(out_path)
        plt.close()
        print(f"t-SNE plot saved to: {out_path}")

    result_lines = [
        "Test Evaluation Results:",
        f"Total queries (retrieval): {total_q}",
        *[f"Hit@{k}: {hit_rates[k]:.2f}%" for k in k_list],
        *[f"Precision@{k}: {precisions[k]:.2f}%" for k in k_list],
        f"Classification Accuracy: {accuracy:.2f}%",
        f"Mean Per Class Accuracy: {mean_per_class_acc:.2f}%",
        f"Confusion Matrix CSV: {csv_path}",
        f"Confusion Matrix JPG: {jpg_path}",
    ]
    for line in result_lines:
        print(line)

    test_log_path = os.path.join(save_path, "test_log.txt")
    with open(test_log_path, "w") as f:
        f.write("\n".join(result_lines))

    return {
        "hit_rates": hit_rates,
        "precisions": precisions,
        "accuracy": accuracy,
        "mean_per_class_accuracy": mean_per_class_acc
    }

def log_metrics_to_wandb(train_log, test_log=None):
    """W&B へのログ出力（train, validate それぞれのログ項目を送信）"""
    if train_log is not None:
        wandb.log({
            "train_loss": train_log['loss'],
            "train_loss_bbox": train_log['loss_bbox'],
            "train_loss_temp_bbox": train_log['loss_temp_bbox'],      
            "train_loss_future_bbox": train_log['loss_future_bbox'],
            "train_loss_flow_spatial": train_log['loss_flow_spatial'],
            "train_loss_flow_temporal": train_log['loss_flow_temporal'],
            "train_loss_supervised": train_log['loss_supervised'],
            "train_epoch_time": train_log['time'],
            "gradient_norm": train_log['grad_norm'],
            "epoch": train_log['epoch'],
        })
    if test_log is not None:
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            wandb.log({
                "val_loss": test_log['loss'],
                "val_loss_bbox": test_log['loss_bbox'],
                "val_loss_temp_bbox": test_log['loss_temp_bbox'],
                "val_loss_future_bbox": test_log['loss_future_bbox'],
                'val_loss_flow_spatial': test_log['loss_flow_spatial'],
                'val_loss_flow_temporal': test_log['loss_flow_temporal'],
                'val_loss_supervised': test_log['loss_supervised'],
                "val_epoch_time": test_log['time'],
                "accuracy": test_log['accuracy'],
                "best_hit_rate": test_log['best_hit_rate'],
                "best_hit_epoch": test_log['best_hit_epoch'],
                "epoch": test_log['epoch'],
            })

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer(object):
    """Simple timer."""
    def __init__(self):
        self.last_time = time.time()
    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time

def initialize_wandb(args):
    if args.dataset == 'volleyball':
        project_name = "flow_ball_volleyball_GAR"
    elif args.dataset == 'nba':
        project_name = "flow_ball_nba_numpy_GAR"
    wandb.init(
        project=project_name,
        name=f'{args.dataset}_experiment_{time.strftime("%Y%m%d-%H%M%S")}',
        config=args,
    )

def train(args, train_loader, model, criterion, ce_criterion, optimizer, epoch, lama_model):
    epoch_timer = Timer()
    losses = AverageMeter()
    losses_bbox = AverageMeter()
    losses_temp_bbox = AverageMeter()
    losses_future_bbox = AverageMeter()
    losses_flow_spatial = AverageMeter()
    losses_flow_temporal = AverageMeter()
    losses_supervised = AverageMeter()
    grad_norms = AverageMeter()
    model.train()
    fine_tune = bool(args.pretrained_weights)
    
    if args.dataset == 'nba':
        mask_size = 40 * args.image_width // 1280
        mask_size_min, mask_size_max = 40 * args.image_width // 1280, 80 * args.image_width // 1280
    elif args.dataset == 'volleyball':
        mask_size = 40 * args.image_width // 1280
        mask_size_min, mask_size_max = 20 * args.image_width // 1280, 60 * args.image_width // 1280
    
    MAX_INPAINT = args.batch * args.num_frame
    buf_imgs   = torch.empty(
        MAX_INPAINT, 3, args.image_height, args.image_width,
        device='cuda', dtype=torch.float32
    )
    buf_coords = torch.empty(MAX_INPAINT, 2, device='cuda', dtype=torch.long)

    for i, sample in enumerate(tqdm(train_loader, desc="Training")):
        loss = torch.tensor(0.0).cuda()
        loss_bbox = torch.tensor(0.0).cuda()
        loss_temp_bbox = torch.tensor(0.0).cuda()
        loss_future_bbox = torch.tensor(0.0).cuda()
        loss_flow_spatial = torch.tensor(0.0).cuda()
        loss_flow_temporal = torch.tensor(0.0).cuda()
        loss_supervised = torch.tensor(0.0).cuda()
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            if args.use_flow or args.use_flow_numpy:
                images, activities, ball_gt, flow, bboxes, _ = sample
                flow = flow.cuda()
            else:
                images, activities, ball_gt, bboxes, _ = sample
            activities = activities.cuda()
            images = images.cuda()
            ball_gt = ball_gt.cuda()
            bboxes = bboxes.cuda()
        B, T, C, H, W = images.shape
        if args.dataset == 'volleyball':
            N = 12
        elif args.dataset == 'nba':
            N = 10
        if args.ball_mask:
            B, T = images.shape[:2]

            ball_gt_real = torch.zeros(B, T, 2, device='cuda', dtype=torch.long)
            ball_gt_real[..., 0] = (ball_gt[..., 0] * args.image_width ).long()
            ball_gt_real[..., 1] = (ball_gt[..., 1] * args.image_height).long()

            imgs_r   = images.view(B * T, 3, args.image_height, args.image_width)
            coords_r = ball_gt_real.view(B * T, 2)

            if args.frame_random:
                mask = torch.rand(B * T, device='cuda') < args.inpaint_prob
            elif args.batch_random:
                batch_mask = [random.random() < args.inpaint_prob for _ in range(B)]
                mask = torch.zeros(B * T, dtype=torch.bool, device='cuda')
                for b in range(B):
                    if batch_mask[b]:
                        mask[b * T : (b + 1) * T] = True
            else:
                mask = torch.ones(B * T, dtype=torch.bool, device='cuda')

            num = int(mask.sum().item())
            if num > 0:
                buf_imgs[:num].copy_(imgs_r[mask])
                buf_coords[:num].copy_(coords_r[mask])
                out = apply_ball_mask_prob(buf_imgs[:num], buf_coords[:num], mask_size=mask_size)

                imgs_r[mask] = out

            images = imgs_r.view(B, T, 3, args.image_height, args.image_width)
        if args.random_mask:
            mask_size_range = (mask_size_min, mask_size_max)
            num_masks_range = (5, 10)
            images = apply_random_mask(images, mask_size_range=mask_size_range, num_masks_range=num_masks_range)
        if args.people_mask:
            imgs_flat   = images.view(B * T, C, H, W)   # (B⋅T, C, H, W)
            bboxes_flat = bboxes.view(B * T, N, 4)      # (B⋅T, N, 4)
            if args.dataset == 'volleyball' or args.dataset == 'nba':
                ori_h, ori_w = 720, 1280
            bboxes_rescaled = bboxes_flat.clone()
            bboxes_rescaled[..., 0] = (bboxes_flat[..., 0] / ori_h) * H
            bboxes_rescaled[..., 2] = (bboxes_flat[..., 2] / ori_h) * H
            bboxes_rescaled[..., 1] = (bboxes_flat[..., 1] / ori_w) * W
            bboxes_rescaled[..., 3] = (bboxes_flat[..., 3] / ori_w) * W
            people_masks, people_masks_flags = generate_people_masks(
                normalized_batch=imgs_flat,
                bboxes=bboxes_rescaled,
                scale=args.people_mask_scale,
                inpaint_prob=args.inpaint_people_prob,
                device='cuda'
            )  # (B⋅T, 1, H, W)
            expanded_mask = people_masks.expand(-1, C, -1, -1)
            imgs_flat[expanded_mask.bool()] = 0.0
            images = imgs_flat.view(B, T, C, H, W)
            
        if args.ball_lama or args.people_lama:

            imgs_flat   = images.view(B * T, C, H, W)   # (B⋅T, C, H, W)
            ball_flat   = ball_gt.view(B * T, 2)   # (B⋅T, 2)
            bboxes_flat = bboxes.view(B * T, N, 4)      # (B⋅T, N, 4)

            if args.ball_lama:
                ball_flat_real = torch.zeros(B * T, 2, device='cuda', dtype=torch.long)
                ball_flat_real[..., 0] = (ball_flat[..., 0] * args.image_width ).long()
                ball_flat_real[..., 1] = (ball_flat[..., 1] * args.image_height).long()

                if args.frame_random:
                    do_ball = (torch.rand(B * T, device='cuda') < args.inpaint_prob)
                elif args.batch_random:
                    batch_mask = [random.random() < args.inpaint_prob for _ in range(B)]
                    do_ball = torch.zeros(B * T, dtype=torch.bool, device='cuda')
                    for b in range(B):
                        if batch_mask[b]:
                            do_ball[b * T : (b + 1) * T] = True
                else:
                    do_ball = torch.ones(B * T, dtype=torch.bool, device='cuda')

                ball_masks = generate_ball_masks(
                    normalized_batch=imgs_flat,
                    centers=ball_flat_real,
                    mask_size=mask_size,
                    mask_shape=args.mask_shape,
                    device='cuda'
                )  # → (B*T, 1, H, W)
                ball_masks[~do_ball] = 0.0
            else:
                ball_masks = torch.zeros(B * T, 1, H, W, device='cuda')

            if args.people_lama:
                if args.dataset == 'volleyball' or args.dataset == 'nba':
                    ori_h, ori_w = 720, 1280
                bboxes_rescaled = bboxes_flat.clone()
                bboxes_rescaled[..., 0] = (bboxes_flat[..., 0] / ori_h) * H
                bboxes_rescaled[..., 2] = (bboxes_flat[..., 2] / ori_h) * H
                bboxes_rescaled[..., 1] = (bboxes_flat[..., 1] / ori_w) * W
                bboxes_rescaled[..., 3] = (bboxes_flat[..., 3] / ori_w) * W

                people_masks, people_masks_flags = generate_people_masks(
                    normalized_batch=imgs_flat,
                    bboxes=bboxes_rescaled,
                    scale=args.people_mask_scale,
                    inpaint_prob=args.inpaint_people_prob,
                    device='cuda'
                )  # (B⋅T, 1, H, W)
            else:
                people_masks = torch.zeros(B * T, 1, H, W, device='cuda')

            combined_masks = torch.clamp(ball_masks + people_masks, 0.0, 1.0)  # (B⋅T, 1, H, W)

            inpainted_flat = inpaint_with_lama_masks(
                normalized_batch=imgs_flat,
                masks=combined_masks,
                lama_model=lama_model,
                device='cuda'
            )  # (B⋅T, C, H, W)

            images = inpainted_flat.view(B, T, C, H, W)
        
        input_data = {'images': images, 'bboxes': bboxes}
        
        if args.ball_pred:
            valid_frame_mask = (ball_gt != 0).any(dim=-1)
            valid_frame_mask_flat = valid_frame_mask.view(-1)
            if not args.flow_pred:
                if valid_frame_mask_flat.any():
                    ret_dic = model(input_data)
                else:
                    continue
            else:
                ret_dic = model(input_data)
        else:
            ret_dic = model(input_data)
        
        if fine_tune:
            if args.dataset == 'volleyball' or args.dataset == 'nba':
                activities_score = ret_dic['activities_score']
                loss_supervised = ce_criterion(activities_score, activities[:, 0])
                loss = loss_supervised
        
        else:
            if args.flow_pred:
                if args.dataset == 'volleyball' or args.dataset == 'nba':
                    if 'people_masks_flags' in locals():
                        masked_flags = people_masks_flags.view(B, T, N).bool()
                    else:
                        masked_flags = torch.zeros((B, T, N), dtype=torch.bool, device='cuda')
                    bboxes_x_center = (((bboxes[:, :, :, 0] + bboxes[:, :, :, 2]) // 2).long())
                    bboxes_y_center = (((bboxes[:, :, :, 1] + bboxes[:, :, :, 3]) // 2).long())
                    batch_indices = torch.arange(B).view(B, 1, 1).expand(B, T, N).cuda()
                    frame_indices = torch.arange(T).view(1, T, 1).expand(B, T, N).cuda()
                    if args.use_flow_numpy:
                        if args.image_width < 896:
                            flow = normalize_flow_minmax(flow)
                            scale_x, scale_y = 0.025, 0.025
                            bboxes_x_center = (bboxes_x_center * scale_x).long()
                            bboxes_y_center = (bboxes_y_center * scale_y).long()
                        else:
                            flow = normalize_flow_minmax(flow)
                            scale_x = args.image_width / 1280
                            scale_y = args.image_height / 720
                            if args.backbone == 'dinov2' or args.backbone == 'franca' or (args.backbone == 'clip' and args.ViT_arch == "vit-l"):
                                bboxes_x_center = ((bboxes_x_center * scale_x) // 14).long()
                                bboxes_y_center = ((bboxes_y_center * scale_y) // 14).long()
                            elif args.backbone == 'dinov3' or args.backbone == 'ViT' or args.backbone == 'MAE' or args.backbone == 'dino' or (args.backbone == 'clip' and args.ViT_arch == "vit-b") or args.backbone == 'siglip' or args.backbone == 'siglip2':
                                bboxes_x_center = ((bboxes_x_center * scale_x) // 16).long()
                                bboxes_y_center = ((bboxes_y_center * scale_y) // 16).long()
                    flow_x = flow[batch_indices, frame_indices, 0, bboxes_y_center, bboxes_x_center]
                    flow_y = flow[batch_indices, frame_indices, 1, bboxes_y_center, bboxes_x_center]
                    flow_gt = torch.stack([flow_x, flow_y], dim=-1)
                    spatial_flow_pred = ret_dic['pred_flow_spatial']
                    temp_flow_pred = ret_dic['pred_flow_temporal']
                    if args.spatial_flow_loss:
                        loss_flow_spatial = weighted_flow_loss_mse(spatial_flow_pred, flow_gt, masked_flags, args.w_inpaint_people)
                        loss += args.w_flow * loss_flow_spatial
                    if args.temporal_flow_loss:
                        loss_flow_temporal = weighted_flow_loss_mse(temp_flow_pred, flow_gt, masked_flags, args.w_inpaint_people)
                        loss += args.w_flow * loss_flow_temporal
            
            if args.ball_pred:
                bbox_pred = ret_dic['pred_bbox_spatial']
                temp_bbox_pred = ret_dic['pred_bbox_temporal']
                bbox_pred_flat = bbox_pred.view(-1, bbox_pred.size(-1))
                temporal_bbox_pred_flat = temp_bbox_pred.view(-1, temp_bbox_pred.size(-1))
                ball_gt_flat = ball_gt.view(-1, ball_gt.size(-1))
                valid_frame_mask_flat = valid_frame_mask.view(-1)
                if valid_frame_mask_flat.any():
                    bbox_pred_valid = bbox_pred_flat[valid_frame_mask_flat]
                    temp_bbox_pred_valid = temporal_bbox_pred_flat[valid_frame_mask_flat]
                    ball_gt_valid = ball_gt_flat[valid_frame_mask_flat]
                    if args.spatial_loss:
                        loss_bbox = criterion(bbox_pred_valid, ball_gt_valid)
                        loss += args.w_ball * loss_bbox
                    if args.temporal_loss:
                        loss_temp_bbox = criterion(temp_bbox_pred_valid, ball_gt_valid)
                        loss += args.w_ball * loss_temp_bbox
                    if args.future_mask:
                        future_bbox_pred = ret_dic['pred_bbox_future']
                        future_bbox_pred = future_bbox_pred.view(B, T, T, -1)
                        future_bbox_pred = future_bbox_pred.permute(1, 0, 2, 3)
                        future_bbox_pred = future_bbox_pred.reshape(T, B * T, -1)
                        future_bbox_pred_valid = future_bbox_pred[:, valid_frame_mask_flat]
                        ball_gt_valid_expand = ball_gt_valid.expand_as(future_bbox_pred_valid)
                        loss_future_bbox = criterion(future_bbox_pred_valid, ball_gt_valid_expand)
                        loss += args.w_ball * loss_future_bbox
                        
            if args.supervised:
                if args.dataset == 'volleyball' or args.dataset == 'nba':
                    activities_score = ret_dic['activities_score']
                    loss_supervised = ce_criterion(activities_score, activities[:, 0])
                    loss += args.w_sup * loss_supervised
        
        if loss.item() == 0:
            loss = loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        else:
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm_sq ** 0.5
        grad_norms.update(grad_norm)
        optimizer.step()
        
        if args.supervised:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
            losses_supervised.update(loss_supervised.item(), B)
        if args.ball_pred and args.flow_pred:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
        elif args.ball_pred:
            losses.update(loss.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
        elif args.flow_pred:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)

    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'loss_bbox': losses_bbox.avg,
        'loss_temp_bbox': losses_temp_bbox.avg,
        'loss_future_bbox': losses_future_bbox.avg,
        'loss_flow_spatial': losses_flow_spatial.avg,
        'loss_flow_temporal': losses_flow_temporal.avg,
        'loss_supervised': losses_supervised.avg,
        'grad_norm': grad_norms.avg,
    }
    log_metrics_to_wandb(train_log, test_log=None)
    return train_log

@torch.no_grad()
def validate(train_loader_for_val, test_loader, model, criterion, ce_criterion, epoch, optimizer, scheduler, save_path):
    global best_hit_rate, best_hit_epoch, best_f1, best_f1_epoch
    epoch_timer = Timer()
    losses = AverageMeter()
    losses_bbox = AverageMeter()
    losses_temp_bbox = AverageMeter()
    losses_future_bbox = AverageMeter()
    losses_flow_spatial = AverageMeter()
    losses_flow_temporal = AverageMeter()
    losses_supervised = AverageMeter()
    correct = 0
    total = 0
    model.eval()
    fine_tune = bool(args.pretrained_weights)

    for sample in tqdm(test_loader, desc="Extracting Test Features"):
        loss = torch.tensor(0.0).cuda()
        loss_bbox = torch.tensor(0.0).cuda()
        loss_temp_bbox = torch.tensor(0.0).cuda()
        loss_future_bbox = torch.tensor(0.0).cuda()
        loss_flow_spatial = torch.tensor(0.0).cuda()
        loss_flow_temporal = torch.tensor(0.0).cuda()
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            if args.use_flow or args.use_flow_numpy:
                images, activities, ball_gt, flow, bboxes, _ = sample
                flow = flow.cuda()
            else:
                images, activities, ball_gt, bboxes, _ = sample
            activities = activities.cuda()
            images = images.cuda()
            ball_gt = ball_gt.cuda()
            bboxes = bboxes.cuda()
        B, T, _, _, _ = images.shape
        if args.dataset == 'volleyball':
            N = 12
        elif args.dataset == 'nba':
            N = 10
        input_data = {'images': images, 'bboxes': bboxes}
        ret_dic = model(input_data)
        
        if fine_tune:
            if args.dataset == 'volleyball' or args.dataset == 'nba':
                activities_score = ret_dic['activities_score']
                loss_supervised = ce_criterion(activities_score, activities[:, 0])
                loss = loss_supervised
                preds = activities_score.argmax(dim=1)
                correct += (preds == activities[:, 0]).sum().item()
                total += activities.size(0)

        else:
            if args.supervised:
                if args.dataset == 'volleyball' or args.dataset == 'nba':
                    activities_score = ret_dic['activities_score']
                    loss_supervised = ce_criterion(activities_score, activities[:, 0])
                    loss += args.w_sup * loss_supervised
                    preds = activities_score.argmax(dim=-1)
                    correct += (preds == activities[:, 0]).sum().item()
                    total += activities.size(0)
            
            if args.flow_pred:
                if args.dataset == 'volleyball' or args.dataset == 'nba':
                    bboxes_x_center = (((bboxes[:, :, :, 0] + bboxes[:, :, :, 2]) // 2).long())
                    bboxes_y_center = (((bboxes[:, :, :, 1] + bboxes[:, :, :, 3]) // 2).long())
                    batch_indices = torch.arange(B).view(B, 1, 1).expand(B, T, N).cuda()
                    frame_indices = torch.arange(T).view(1, T, 1).expand(B, T, N).cuda()
                    if args.use_flow_numpy:
                        if args.image_width < 896:
                            flow = normalize_flow_minmax(flow)
                            scale_x, scale_y = 0.025, 0.025
                            bboxes_x_center = (bboxes_x_center * scale_x).long()
                            bboxes_y_center = (bboxes_y_center * scale_y).long()
                        else:
                            flow = normalize_flow_minmax(flow)
                            scale_x = args.image_width / 1280
                            scale_y = args.image_height / 720
                            if args.backbone == 'dinov2' or args.backbone == 'franca' or (args.backbone == 'clip' and args.ViT_arch == "vit-l"):
                                bboxes_x_center = ((bboxes_x_center * scale_x) // 14).long()
                                bboxes_y_center = ((bboxes_y_center * scale_y) // 14).long()
                            elif args.backbone == 'dinov3' or args.backbone == 'ViT' or args.backbone == 'MAE' or args.backbone == 'dino' or (args.backbone == 'clip' and args.ViT_arch == "vit-b") or args.backbone == 'siglip' or args.backbone == 'siglip2':
                                bboxes_x_center = ((bboxes_x_center * scale_x) // 16).long()
                                bboxes_y_center = ((bboxes_y_center * scale_y) // 16).long()
                    flow_x = flow[batch_indices, frame_indices, 0, bboxes_y_center, bboxes_x_center]
                    flow_y = flow[batch_indices, frame_indices, 1, bboxes_y_center, bboxes_x_center]
                    flow_gt = torch.stack([flow_x, flow_y], dim=-1)
                    spatial_flow_pred = ret_dic['pred_flow_spatial']
                    temp_flow_pred = ret_dic['pred_flow_temporal']
                    if args.spatial_flow_loss:
                        loss_flow_spatial = criterion(spatial_flow_pred, flow_gt)
                        loss += args.w_flow * loss_flow_spatial
                    if args.temporal_flow_loss:
                        loss_flow_temporal = criterion(temp_flow_pred, flow_gt)
                        loss += args.w_flow * loss_flow_temporal
            
            if args.ball_pred:
                bbox_pred = ret_dic['pred_bbox_spatial']
                temp_bbox_pred = ret_dic['pred_bbox_temporal']
                valid_frame_mask = (ball_gt != 0).any(dim=-1)
                bbox_pred_flat = bbox_pred.view(-1, bbox_pred.size(-1))
                temporal_bbox_pred_flat = temp_bbox_pred.view(-1, temp_bbox_pred.size(-1))
                ball_gt_flat = ball_gt.view(-1, ball_gt.size(-1))
                valid_frame_mask_flat = valid_frame_mask.view(-1)
                if valid_frame_mask_flat.any():
                    bbox_pred_valid = bbox_pred_flat[valid_frame_mask_flat]
                    temp_bbox_pred_valid = temporal_bbox_pred_flat[valid_frame_mask_flat]
                    ball_gt_valid = ball_gt_flat[valid_frame_mask_flat]
                    if args.spatial_loss:
                        loss_bbox = criterion(bbox_pred_valid, ball_gt_valid)
                        loss += args.w_ball * loss_bbox
                    if args.temporal_loss:
                        loss_temp_bbox = criterion(temp_bbox_pred_valid, ball_gt_valid)
                        loss += args.w_ball * loss_temp_bbox
                    if args.future_mask:
                        future_bbox_pred = ret_dic['pred_bbox_future']
                        future_bbox_pred = future_bbox_pred.view(B, T, T, -1)
                        future_bbox_pred = future_bbox_pred.permute(1, 0, 2, 3)
                        future_bbox_pred = future_bbox_pred.reshape(T, B * T, -1)
                        future_bbox_pred_valid = future_bbox_pred[:, valid_frame_mask_flat]
                        ball_gt_valid_expand = ball_gt_valid.expand_as(future_bbox_pred_valid)
                        loss_future_bbox = criterion(future_bbox_pred_valid, ball_gt_valid_expand)
                        loss += args.w_ball * loss_future_bbox
        
        video_features_test = ret_dic['video_features']
            
        if args.supervised:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
            losses_supervised.update(loss_supervised.item(), B)
        elif args.ball_pred and args.flow_pred:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
        elif args.ball_pred:
            losses.update(loss.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
        elif args.flow_pred:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)

    if args.dataset == 'volleyball' or args.dataset == 'nba':
        accuracy = correct / total * 100
        if accuracy > best_hit_rate:
            best_hit_rate  = accuracy
            best_hit_epoch = epoch
            save_best_model(epoch, model, optimizer, scheduler, accuracy, save_path)

    if args.dataset == 'volleyball' or args.dataset == 'nba':
        val_log = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            'loss': losses.avg,
            'loss_bbox': losses_bbox.avg,
            'loss_temp_bbox': losses_temp_bbox.avg,
            'loss_future_bbox': losses_future_bbox.avg,
            'loss_flow_spatial': losses_flow_spatial.avg,
            'loss_flow_temporal': losses_flow_temporal.avg,
            'loss_supervised': losses_supervised.avg,
            'accuracy': accuracy,
            'best_hit_rate': best_hit_rate,
            'best_hit_epoch': best_hit_epoch,
        }
    log_metrics_to_wandb(train_log=None, test_log=val_log)
    return val_log

def worker_init_fn(worker_id):
    seed = args.random_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)

def main():
    global args, best_hit_rate, best_hit_epoch, best_model_path

    initialize_wandb(args)
    
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_dino_flow_ball_GAR<%s>' % (args.dataset, time_str)
    save_path = './GAR_result/%s' % exp_name
    os.makedirs(save_path, exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == 'volleyball' or args.dataset == 'nba':
        train_set, train_set_for_val, test_set, _, _, _, _, _ = read_dataset(args)
    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    train_loader_for_val = data.DataLoader(train_set_for_val, batch_size=args.feature_batch, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    model = models.Ball_detect_model(args)
    model = torch.nn.DataParallel(model).cuda()
    
    if args.pretrained_weights:
        checkpoint = torch.load(args.pretrained_weights, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded pretrained weights from {args.pretrained_weights}, entering fine-tune mode.")
    
    if args.ball_lama or args.people_lama:
        lama_model = load_lama_model(args.lama_model_dir, checkpoint_name=args.checkpoint_name, device='cuda')
    else:
        lama_model = None

    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    criterion = nn.MSELoss().cuda()

    if args.dataset == 'volleyball' or args.dataset == 'nba':
        ce_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    start_epoch = 1
    
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- Train at epoch #%d' % epoch)
        train_log = train(args, train_loader, model, criterion, ce_criterion, optimizer, epoch, lama_model)
        print_log(save_path, 'Train - Epoch: %d, Loss: %.4f, Time: %.1f sec' % (epoch, train_log['loss'], train_log['time']))
        current_lr = scheduler.get_last_lr()[0]
        print('Current learning rate: %f' % current_lr)
        wandb.log({"learning_rate": current_lr, "epoch": epoch})
        scheduler.step()

        if epoch % args.test_freq == 0:
            print_log(save_path, '----- Validate at epoch #%d' % epoch)
            val_log = validate(train_loader_for_val, test_loader, model, criterion, ce_criterion, epoch, optimizer, scheduler, save_path)
            if args.dataset == 'volleyball' or args.dataset == 'nba':
                print_log(save_path, 'Validation - Epoch: %d, Loss: %.4f, Time: %.1f sec, accuracy: %.2f%%' %
                        (epoch, val_log['loss'], val_log['time'], val_log['accuracy']))
                print_log(save_path, 'Best accuracy: %.2f%% at epoch #%d.' % (val_log['best_hit_rate'], val_log['best_hit_epoch']))

    if best_model_path is not None:
        print("Training complete. Loading best model from:", best_model_path)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            test_evaluation(train_loader_for_val, test_loader, model, args, save_path)
    else:
        print("No best model was saved.")

if __name__ == '__main__':
    main()
