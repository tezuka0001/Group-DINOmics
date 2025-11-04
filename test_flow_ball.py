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
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import NearestNeighbors
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as mpatheffects

import models.models_flow_ball as models
# import models.models_flow_ball_fix_dino as models
from util.utils import *
from dataloader.dataloader_bbox_flow_detector import read_dataset
from sklearn.manifold import TSNE
from tqdm import tqdm

# import lama
from omegaconf import OmegaConf
import yaml
from lama.bin.predict_inpaint import *
import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

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
parser.add_argument('--nheads_agg', default=4, type=int, help='number of heads for partial context aggregation')

# GPU
parser.add_argument('--device', default="0", type=str, help='GPU device')

# Load model
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

parser.add_argument('--hidden_size', default=1024, type=int, help='hidden size')
parser.add_argument('--detector', action='store_true', help='use detector')

# backbone parameters
parser.add_argument('--backbone_learnable', action='store_true', help='use learnable last layer')
parser.add_argument('--backbone_full_learnable', action='store_true', help='use full learnable last layer')
parser.add_argument('--backbone_learnable_layers', default=1, type=int, help='number of learnable layers')
parser.add_argument('--backbone', default='dinov3', type=str, help='backbone model, dinov2, clip, ViT, MAE, resnet50, vgg16, vgg19')
parser.add_argument('--ViT_arch', default='vit-l', type=str, help='vit-l, vit-b')
parser.add_argument('--use_lora', action='store_true', help='use LoRA for backbone')
parser.add_argument('--lora_blocks', nargs='+', type=int, default=list(range(12, 22)), help='LoRA blocks to apply')
parser.add_argument('--linear_probing', action='store_true', help='use linear probing for backbone')
parser.add_argument('--spatial_backbone_mlp', action='store_true', help='use spatial mlp for backbone')
parser.add_argument('--spatial_mlp_flow', action='store_true', help='use spatial flow mlp for backbone')
parser.add_argument('--spatial_mlp_ball', action='store_true', help='use spatial mlp for ball for backbone')
parser.add_argument('--temp_mlp_flow', action='store_true', help='use temporal flow mlp for backbone')

parser.add_argument('--ball_mask', action='store_true', help='use ball mask')
parser.add_argument('--random_mask', action='store_true', help='use random mask')
parser.add_argument('--ball_inpaint', action='store_true', help='use ball inpaint')
parser.add_argument('--people_mask', action='store_true', help='use people mask')
parser.add_argument('--ball_pred', action='store_true', help='use ball prediction')
parser.add_argument('--flow_pred', action='store_true', help='use flow prediction')

parser.add_argument('--supervised', action='store_true', help='use supervised learning')
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

args = parser.parse_args()
args.use_flow = False
args.use_flow_numpy = args.flow_pred or args.flow_recon

def _save_tsne_with_text(xs, ys, labels, colors, out_path,
                         title="t-SNE (with IDs)", fontsize=6, alpha=0.9):
    fig, ax = plt.subplots(figsize=(16, 12))
    sc = ax.scatter(xs, ys, c=colors, cmap='tab10', s=12, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    try:
        plt.colorbar(sc, ax=ax)
    except Exception:
        pass

    text_effect = [mpatheffects.withStroke(linewidth=2, foreground="white")]
    for x, y, txt in zip(xs, ys, labels):
        t = ax.text(x, y, txt, fontsize=fontsize, alpha=alpha)
        t.set_path_effects(text_effect)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"t-SNE with IDs saved to: {out_path}")

@torch.no_grad()
def test_evaluation(train_loader_for_val, test_loader, model, args, save_path, train_frames, test_frames, lama_model):
    model.eval()
        
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
    
    train_features_list = []
    train_labels_list = []
    for sample in tqdm(train_loader_for_val, desc="Extracting Train Features for Test"):
        if args.use_flow or args.use_flow_numpy:
            images, activities, ball_gt, _, bboxes, _ = sample
        else:
            images, activities, ball_gt, bboxes, _ = sample
        images = images.cuda()
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
                out = apply_ball_mask_circle(buf_imgs[:num], buf_coords[:num], mask_size=mask_size)

                imgs_r[mask] = out

            images = imgs_r.view(B, T, 3, args.image_height, args.image_width)
        if args.random_mask:
            mask_size_range = (mask_size_min, mask_size_max)
            num_masks_range = (5, 10)
            images = apply_random_mask(images, mask_size_range=mask_size_range, num_masks_range=num_masks_range)
        if args.people_mask:
            imgs_flat   = images.view(B * T, C, H, W)   # (B⋅T, C, H, W)
            bboxes_flat = bboxes.view(B * T, N, 4)      # (B⋅T, N, 4)
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
        ret_dic = model(input_data)
        video_features = ret_dic['video_features']
        train_features_list.append(video_features.cpu().numpy())
        train_labels_list.append(activities[:, 0].cpu().numpy())
    train_features = np.vstack(train_features_list)
    train_labels = np.hstack(train_labels_list)

    test_features_list = []
    test_labels_list = []
    for sample in tqdm(test_loader, desc="Extracting Test Features"):
        if args.use_flow or args.use_flow_numpy:
            images, activities, ball_gt, flow, bboxes, _ = sample
        else:
            images, activities, ball_gt, bboxes, _ = sample
        images = images.cuda()

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
                out = apply_ball_mask_circle(buf_imgs[:num], buf_coords[:num], mask_size=mask_size)

                imgs_r[mask] = out

            images = imgs_r.view(B, T, 3, args.image_height, args.image_width)
        if args.random_mask:
            mask_size_range = (mask_size_min, mask_size_max)
            num_masks_range = (5, 10)
            images = apply_random_mask(images, mask_size_range=mask_size_range, num_masks_range=num_masks_range)
        if args.people_mask:
            imgs_flat   = images.view(B * T, C, H, W)   # (B⋅T, C, H, W)
            bboxes_flat = bboxes.view(B * T, N, 4)      # (B⋅T, N, 4)
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
        ret_dic = model(input_data)
        video_features = ret_dic['video_features']
        test_features_list.append(video_features.cpu().numpy())
        test_labels_list.append(activities[:, 0].cpu().numpy())
    test_features = np.vstack(test_features_list)
    test_labels = np.hstack(test_labels_list)
    
    k_list = [1, 2, 3, 4, 5]
    nbrs_k = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(train_features)
    distances_k, indices_k = nbrs_k.kneighbors(test_features)

    if args.dataset == 'volleyball':
        label_names = [
            'R-set', 'R-spike', 'R-pass', 'R-winpoint',
            'L-set', 'L-spike', 'L-pass', 'L-winpoint'
        ]
    elif args.dataset == 'nba':
        label_names = [
            '2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
            '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
            '3p-succ.', '3p-fail.-off.', '3p-fail.-def.'
        ]

    hit1_ids_path = os.path.join(save_path, "hit1_ids.txt")
    with open(hit1_ids_path, "w") as f_id:
        for i, neighbors in enumerate(indices_k):
            retrieved_idx = neighbors[0]
            query_vid, query_fr = test_frames[i]
            query_label = label_names[test_labels[i]]
            ret_vid, ret_fr = train_frames[retrieved_idx]
            ret_label = label_names[train_labels[retrieved_idx]]

            f_id.write(
                f"({query_vid}, {query_fr}), {query_label}, "
                f"({ret_vid}, {ret_fr}), {ret_label}\n"
            )
    print(f"hit@1 IDs saved to {hit1_ids_path}.")

    predicted_labels = train_labels[indices_k[:, 0]]
    cm = confusion_matrix(test_labels, predicted_labels)
    cm_df = pd.DataFrame(cm)
    csv_path = os.path.join(save_path, "confusion_matrix.csv")
    cm_df.to_csv(csv_path, index=False)
    
    plot_confusion_matrices_from_dir(args, save_path)
    
    jpg_path = os.path.join(save_path, "confusion_matrix.jpg")

    hit_counts = {k: 0 for k in k_list}
    precision_sum = {k: 0.0 for k in k_list}
    for i, neighbors in enumerate(indices_k):
        neighbor_labels = train_labels[neighbors]  # shape: (5,)
        for k in k_list:
            correct_count = np.sum(neighbor_labels[:k] == test_labels[i])
            precision_sum[k] += correct_count / k
            if correct_count > 0:
                hit_counts[k] += 1

    total_queries = len(test_labels)
    hit_rates = {k: hit_counts[k] / total_queries * 100.0 for k in k_list}
    precisions = {k: precision_sum[k] / total_queries * 100.0 for k in k_list}

    tsne = TSNE(n_components=2, random_state=42)
    tsne_train_results = tsne.fit_transform(train_features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_train_results[:, 0], tsne_train_results[:, 1], c=train_labels, cmap='tab10', s=20)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Train Feature Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    train_tsne_path = os.path.join(save_path, "feature_space_train_tsne.jpg")
    plt.savefig(train_tsne_path)
    plt.close()
    print(f"t-SNE plot saved to: {train_tsne_path}")

    if len(train_frames) == len(train_features):
        train_text_labels = [f"{vid}:{fr}" for (vid, fr) in train_frames]

        train_ids_tsne_path = os.path.join(save_path, "feature_space_train_tsne_ids.jpg")
        _save_tsne_with_text(
            tsne_train_results[:, 0], tsne_train_results[:, 1],
            labels=train_text_labels,
            colors=train_labels,
            out_path=train_ids_tsne_path,
            title="t-SNE of Train Feature Space (IDs)",
            fontsize=4,
            alpha=0.7
        )
    else:
        print("[Warn] Number of train_frames does not match number of train_features, skipping ID rendering.")

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(test_features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=test_labels, cmap='tab10', s=20)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Test Feature Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    tsne_path = os.path.join(save_path, "feature_space_tsne.jpg")
    plt.savefig(tsne_path)
    plt.close()
    print(f"t-SNE plot saved to: {tsne_path}")
    
    if len(test_frames) == len(test_features):
        test_text_labels = [f"{vid}:{fr}" for (vid, fr) in test_frames]

        test_ids_tsne_path = os.path.join(save_path, "feature_space_tsne_ids.jpg")
        _save_tsne_with_text(
            tsne_results[:, 0], tsne_results[:, 1],
            labels=test_text_labels,
            colors=test_labels,
            out_path=test_ids_tsne_path,
            title="t-SNE of Test Feature Space (IDs)",
            fontsize=4,
            alpha=0.7
        )
    else:
        print("[Warn] Number of test_frames does not match number of test_features, skipping ID rendering.")

    result_lines = []
    result_lines.append("Test Evaluation Results:")
    result_lines.append("Total queries: {}".format(total_queries))
    for k in k_list:
        result_lines.append("Hit@{}: {:.2f}%".format(k, hit_rates[k]))
        result_lines.append("Precision@{}: {:.2f}%".format(k, precisions[k]))
    result_lines.append("Confusion Matrix CSV: {}".format(csv_path))
    result_lines.append("Confusion Matrix JPG: {}".format(jpg_path))
    result_lines.append("t-SNE plot: {}".format(tsne_path))

    for line in result_lines:
        print(line)

    test_log_path = os.path.join(save_path, "test_log.txt")
    with open(test_log_path, "w") as f:
        for line in result_lines:
            f.write(line + "\n")

def main():
    global args
    
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = f"[{args.dataset}]_flow_dino_Test_{time_str}"
    save_path = os.path.join('./test_result', exp_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved in: {save_path}")

    print_log(save_path, "Loading dataset...")

    if args.dataset == 'volleyball' or args.dataset == 'nba':
        _, train_set_for_val, test_set, _, _, _, train_frames, test_frames = read_dataset(args)
    train_loader_for_val = data.DataLoader(train_set_for_val, batch_size=args.test_batch, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=2, pin_memory=True)

    print_log(save_path, "Loading model...")
    model = models.Ball_detect_model(args)
    model = torch.nn.DataParallel(model).cuda()
    
    if args.ball_lama or args.people_lama:
        lama_model = load_lama_model(args.lama_model_dir, checkpoint_name=args.checkpoint_name, device='cuda')
    else:
        lama_model = None

    # checkpoint = torch.load(args.model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)

    print_log(save_path, "Evaluating model...")
    if args.dataset == 'volleyball' or args.dataset == 'nba':
        test_evaluation(train_loader_for_val, test_loader, model, args, save_path, train_frames, test_frames, lama_model)

if __name__ == "__main__":
    main()
