import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import Normalize


def print_log(result_path, *args):
    os.makedirs(result_path, exist_ok=True)

    print(*args)
    file_path = result_path + '/log.txt'
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)
            
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def apply_ball_mask_circle(images, bbox_gt, mask_size=20):
    B, C, H, W = images.shape
    device = images.device
    mask_radius = mask_size // 2

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )  # shape (H, W)

    for b in range(B):
        cx, cy = bbox_gt[b]  # center coordinates (x,y)
        cx, cy = int(cx.item()), int(cy.item())

        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist2 <= mask_radius ** 2  # True: inside the circle

        images[b, :, mask] = 0

    return images

def apply_random_mask(images, mask_size_range=(20, 100), num_masks_range=(1, 5)):
    B, T, C, H, W = images.shape
    for b in range(B):
        for t in range(T):
            n_masks = torch.randint(num_masks_range[0], num_masks_range[1] + 1, (1,)).item()
            for _ in range(n_masks):
                mask_size = torch.randint(mask_size_range[0], mask_size_range[1] + 1, (1,)).item()
                if W - mask_size > 0:
                    x1 = torch.randint(0, W - mask_size + 1, (1,)).item()
                else:
                    x1 = 0
                if H - mask_size > 0:
                    y1 = torch.randint(0, H - mask_size + 1, (1,)).item()
                else:
                    y1 = 0
                images[b, t, :, y1:y1+mask_size, x1:x1+mask_size] = 0
    return images

def plot_confusion_matrices_from_dir(args, dir_path):
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_path = os.path.join(dir_path, csv_file)
        file_name = os.path.splitext(csv_file)[0]
        output_path = os.path.join(dir_path, f"{file_name}.jpg")
        
        df = pd.read_csv(csv_path, header=None, skiprows=1)
        data_vals = df.values
        
        row_sums = data_vals.sum(axis=1, keepdims=True)
        normalized_data = data_vals / row_sums
        
        if args.dataset == 'volleyball':
            if args.num_activities == 8:
                labels = ['R-set', 'R-spike', 'R-pass', 'R-winpoint', 
                        'L-set', 'L-spike', 'L-pass', 'L-winpoint']
            elif args.num_activities == 6:
                labels = ['R-set&pass', 'R-spike', 'R-winpoint', 
                        'L-set&pass', 'L-spike', 'L-winpoint']
        elif args.dataset == 'nba':
            labels = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(normalized_data, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=False, yticklabels=labels, vmin=0, vmax=1, square=True, ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Ground-truth')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to: {output_path}")

def normalize_flow_minmax(flow: torch.Tensor) -> torch.Tensor:
    B, T, H, W, _ = flow.shape

    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # flatten (B, T, H*W)
    flow_x_flat = flow_x.view(B, T, -1)
    flow_y_flat = flow_y.view(B, T, -1)

    # min, max (B, T, 1)
    min_x = flow_x_flat.min(dim=-1, keepdim=True)[0]
    max_x = flow_x_flat.max(dim=-1, keepdim=True)[0]
    min_y = flow_y_flat.min(dim=-1, keepdim=True)[0]
    max_y = flow_y_flat.max(dim=-1, keepdim=True)[0]

    flow_x_norm_flat = (flow_x_flat - min_x) / (max_x - min_x + 1e-5)
    flow_y_norm_flat = (flow_y_flat - min_y) / (max_y - min_y + 1e-5)

    flow_x_norm = flow_x_norm_flat.view(B, T, H, W)
    flow_y_norm = flow_y_norm_flat.view(B, T, H, W)

    flow_norm = torch.stack([flow_x_norm, flow_y_norm], dim=2)

    return flow_norm

def calculate_hit_and_precision(train_features, train_labels, query_features, query_labels, k_list):
    max_k = max(k_list)

    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='brute').fit(train_features)
    distances, indices = nbrs.kneighbors(query_features)  # [num_query, max_k]

    neighbor_labels = train_labels[indices]  # shape: [N_query, max_k]
    query_labels_expanded = query_labels[:, np.newaxis]  # shape: [N_query, 1]

    hit_rates = {}
    precisions = {}
    for k in k_list:
        topk = neighbor_labels[:, :k]  # shape: [N_query, k]
        correct_matches = (topk == query_labels_expanded)  # shape: [N_query, k]
        hit_flags = correct_matches.any(axis=1)  # shape: [N_query]
        precision_scores = correct_matches.sum(axis=1) / k  # shape: [N_query]
        hit_rates[k] = hit_flags.mean() * 100.0
        precisions[k] = precision_scores.mean() * 100.0

    return hit_rates, precisions

def weighted_flow_loss_mse(
    flow_pred,          # Tensor, shape (B, T, N, 2)
    flow_gt,            # Tensor, shape (B, T, N, 2)
    masked_flags,       # BoolTensor, shape (B, T, N)
    w_inpaint_people    # float
):
    per_inst = (flow_pred - flow_gt).pow(2)
    per_inst = per_inst.mean(dim=-1)

    weights = 1.0 + w_inpaint_people * masked_flags.float()

    loss = (per_inst * weights).sum() / (weights.sum() + 1e-6)
    return loss

def add_bbox_center_noise_pixels(
    bboxes: torch.Tensor,   # (B, T, N, 4) in pixel coords: [x1, y1, x2, y2]
    sigma_c: float,         # center jitter std in pixels (e.g., 5.0)
    img_w: int=1280,
    img_h: int=720,
    min_wh: float = 1.0,    # minimum width/height in pixels
):
    """
    Add Gaussian noise to bbox centers only (keep bbox size).
    - Skips invalid boxes (assumed to be [0,0,0,0]).
    - Clamps to image bounds and keeps x1<x2, y1<y2.
    """
    noisy = bboxes.clone()

    # invalid if all zeros
    valid = (bboxes.abs().sum(dim=-1) > 0)  # (B,T,N) bool

    x1 = noisy[..., 0]
    y1 = noisy[..., 1]
    x2 = noisy[..., 2]
    y2 = noisy[..., 3]

    # size (kept)
    w = (x2 - x1).clamp(min=min_wh)
    h = (y2 - y1).clamp(min=min_wh)

    # center
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # center noise (only for valid)
    dcx = torch.randn_like(cx) * sigma_c
    dcy = torch.randn_like(cy) * sigma_c
    dcx = dcx * valid.to(noisy.dtype)
    dcy = dcy * valid.to(noisy.dtype)

    cx = (cx + dcx).clamp(0.0, float(img_w - 1))
    cy = (cy + dcy).clamp(0.0, float(img_h - 1))

    # reconstruct bbox with same size
    x1n = (cx - w / 2.0)
    x2n = (cx + w / 2.0)
    y1n = (cy - h / 2.0)
    y2n = (cy + h / 2.0)

    # clamp to bounds
    x1n.clamp_(0.0, float(img_w - 1))
    x2n.clamp_(0.0, float(img_w - 1))
    y1n.clamp_(0.0, float(img_h - 1))
    y2n.clamp_(0.0, float(img_h - 1))

    # ensure ordering + min size again (in case of boundary clipping)
    x1n2 = torch.minimum(x1n, x2n - min_wh)
    x2n2 = torch.maximum(x2n, x1n2 + min_wh)
    y1n2 = torch.minimum(y1n, y2n - min_wh)
    y2n2 = torch.maximum(y2n, y1n2 + min_wh)

    # final clamp
    x1n2.clamp_(0.0, float(img_w - 1))
    x2n2.clamp_(0.0, float(img_w - 1))
    y1n2.clamp_(0.0, float(img_h - 1))
    y2n2.clamp_(0.0, float(img_h - 1))

    out = noisy
    out[..., 0] = x1n2
    out[..., 1] = y1n2
    out[..., 2] = x2n2
    out[..., 3] = y2n2

    # keep invalid boxes exactly zero
    out[~valid] = 0.0
    return out

def add_ball_noise_px_skip_zeros(
    ball_gt: torch.Tensor,     # (B,T,2) normalized in [0,1], (0,0)=missing
    sigma_px: float = 5.0,     # std in pixels (same for x,y)
    H: int = 720,
    W: int = 1280,
):
    valid = (ball_gt[..., 0] != 0) | (ball_gt[..., 1] != 0)  # (B,T)

    # px -> normalized std
    sigma_x = sigma_px / float(W)
    sigma_y = sigma_px / float(H)

    noise = torch.randn_like(ball_gt)
    noise[..., 0] *= sigma_x
    noise[..., 1] *= sigma_y
    noise = noise * valid[..., None].to(ball_gt.dtype)

    noisy = (ball_gt + noise).clamp(0.0, 1.0)
    noisy[~valid] = 0.0
    return noisy