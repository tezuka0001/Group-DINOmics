import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
import yaml

import os
import torch
import torch.nn.functional as F
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
import yaml

def pad_to_modulo(tensor, modulo=8):
    h, w = tensor.shape[-2:]
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

def load_lama_model(model_dir, checkpoint_name='model.pth', device='cuda'):
    config_path = os.path.join(model_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(model_dir, 'models', checkpoint_name)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
    model.to(device)
    model.eval()
    model.freeze()
    return model

@torch.no_grad()
def inpaint_with_lama_batch(normalized_batch, centers, mask_size, lama_model, device='cuda', mask_shape='circle'):
    B, C, H, W = normalized_batch.shape
    normalized_batch = normalized_batch.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, C, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, C, 1, 1)
    images = normalized_batch * std + mean

    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    xx = xx.unsqueeze(0).expand(B, -1, -1)

    if isinstance(centers, list):
        centers = torch.tensor(centers, device=device)
    cx = centers[:, 0].view(B, 1, 1)
    cy = centers[:, 1].view(B, 1, 1)

    if mask_shape == 'circle':
        masks = ((xx - cx)**2 + (yy - cy)**2 <= (mask_size / 2)**2).float()
    elif mask_shape == 'square':
        half = mask_size // 2
        x_in = (xx >= (cx - half)) & (xx <= (cx + half))
        y_in = (yy >= (cy - half)) & (yy <= (cy + half))
        masks = (x_in & y_in).float()
    else:
        raise ValueError("mask_shape must be 'circle' or 'square'")

    masks = masks.unsqueeze(1)

    images_padded = pad_to_modulo(images, 8)
    masks_padded = pad_to_modulo(masks, 8)

    batch = {
        'image': images_padded,
        'mask': masks_padded
    }
    output = lama_model(batch)
    inpainted = output['inpainted'].clamp(0, 1)
    inpainted_cropped = inpainted[:, :, :H, :W]

    normalized_output = (inpainted_cropped - mean) / std
    return normalized_output

def generate_ball_masks(normalized_batch, centers, mask_size, mask_shape, device='cuda'):
    B2, C2, H2, W2 = normalized_batch.shape
    yy, xx = torch.meshgrid(
        torch.arange(H2, device=device),
        torch.arange(W2, device=device),
        indexing='ij'
    )
    yy = yy.unsqueeze(0).expand(B2, -1, -1)
    xx = xx.unsqueeze(0).expand(B2, -1, -1)

    if isinstance(centers, list):
        centers = torch.tensor(centers, device=device)
    cx = centers[:, 0].view(B2, 1, 1)
    cy = centers[:, 1].view(B2, 1, 1)

    if mask_shape == 'circle':
        r = mask_size / 2
        masks = ((xx - cx)**2 + (yy - cy)**2 <= r**2).float()
    elif mask_shape == 'square':
        half = mask_size // 2
        x_in = (xx >= (cx - half)) & (xx <= (cx + half))
        y_in = (yy >= (cy - half)) & (yy <= (cy + half))
        masks = (x_in & y_in).float()
    else:
        raise ValueError("mask_shape must be 'circle' or 'square'")
    return masks.unsqueeze(1)  # (B2, 1, H2, W2)

def generate_people_masks(
    normalized_batch,
    bboxes,
    scale=1.0,
    inpaint_prob=1.0,
    device='cuda'
):
    B2, C, H, W = normalized_batch.shape
    _, N, _    = bboxes.shape

    masks = torch.zeros(B2, 1, H, W, device=device, dtype=torch.float32)
    masked_flags = torch.zeros(B2, N, device=device, dtype=torch.bool)

    for i in range(B2):
        for j in range(N):
            if torch.rand(1).item() >= inpaint_prob:
                continue  # no inpaint for this person

            x1f, y1f, x2f, y2f = bboxes[i, j].tolist()
            cx = (x1f + x2f) / 2.0
            cy = (y1f + y2f) / 2.0
            w0 = (x2f - x1f)
            h0 = (y2f - y1f)

            w_scaled = w0 * scale
            h_scaled = h0 * scale

            x1s = cx - w_scaled / 2.0
            x2s = cx + w_scaled / 2.0
            y1s = cy - h_scaled / 2.0
            y2s = cy + h_scaled / 2.0

            x1 = max(0, min(int(x1s), W))
            x2 = max(0, min(int(x2s), W))
            y1 = max(0, min(int(y1s), H))
            y2 = max(0, min(int(y2s), H))

            if x2 > x1 and y2 > y1:
                masks[i, 0, y1:y2, x1:x2] = 1.0
                masked_flags[i, j] = True  # mark as masked

    return masks, masked_flags  # (B2, 1, H, W)

def generate_net_masks(
    normalized_batch,   
    net_bboxes,         
    scale=1.0,
    bbox_normalized=True,
    device='cuda',
    extend_to_top: bool = False,
    extend_side: str = 'none',   # 'none' | 'auto' | 'left' | 'right' | 'both'
):
    B2, C, H, W = normalized_batch.shape
    net_bboxes_flat = net_bboxes.view(B2, 4).to(device)  # (B2, 4)

    net_masks = torch.zeros(B2, 1, H, W, device=device, dtype=torch.float32)

    extend_side = extend_side.lower()
    if extend_side not in ('none', 'auto', 'left', 'right', 'both'):
        raise ValueError(f"extend_side must be one of "
                         f"['none','auto','left','right','both'], got: {extend_side}")

    for i in range(B2):
        x1f = net_bboxes_flat[i, 0].item()
        y1f = net_bboxes_flat[i, 1].item()
        x2f = net_bboxes_flat[i, 2].item()
        y2f = net_bboxes_flat[i, 3].item()

        if bbox_normalized:
            x1f *= W
            x2f *= W
            y1f *= H
            y2f *= H

        cx = (x1f + x2f) / 2.0
        cy = (y1f + y2f) / 2.0
        w0 = (x2f - x1f)
        h0 = (y2f - y1f)

        if w0 <= 0 or h0 <= 0:
            continue

        w_scaled = w0 * scale
        h_scaled = h0 * scale

        x1s = cx - w_scaled / 2.0
        x2s = cx + w_scaled / 2.0
        y1s = cy - h_scaled / 2.0
        y2s = cy + h_scaled / 2.0

        if extend_to_top:
            y1s = 0.0

        if extend_side == 'both':
            x1s = 0.0
            x2s = float(W)
        elif extend_side == 'left':
            x1s = 0.0
        elif extend_side == 'right':
            x2s = float(W)
        elif extend_side == 'auto':
            if cx < (W / 2.0):
                x1s = 0.0
            else:
                x2s = float(W)

        x1 = max(0, min(int(x1s), W))
        x2 = max(0, min(int(x2s), W))
        y1 = max(0, min(int(y1s), H))
        y2 = max(0, min(int(y2s), H))

        if x2 > x1 and y2 > y1:
            net_masks[i, 0, y1:y2, x1:x2] = 1.0

    return net_masks

def generate_volleyball_net_masks(
    normalized_batch: torch.Tensor,   # (B2, C, H, W)
    net_bboxes: torch.Tensor,         # (B2, 4), [xmin,ymin,xmax,ymax]
    scale: float = 1.0,
    bbox_normalized: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    assert normalized_batch.dim() == 4, "normalized_batch must be (B2, C, H, W)"
    B2, _, H, W = normalized_batch.shape# (B2, C, H, W)

    if device is None:
        device = normalized_batch.device
    else:
        device = torch.device(device)

    net_bboxes_flat = net_bboxes.reshape(-1, 4)  # (B2, 4)

    net_bboxes_flat = net_bboxes_flat.to(device=device, dtype=torch.float32)

    net_masks = torch.zeros((B2, 1, H, W), device=device, dtype=torch.float32)

    for i in range(B2):
        xmin = net_bboxes_flat[i, 0].item()
        ymin = net_bboxes_flat[i, 1].item()
        xmax = net_bboxes_flat[i, 2].item()
        ymax = net_bboxes_flat[i, 3].item()

        if bbox_normalized:
            xmin *= W
            xmax *= W
            ymin *= H
            ymax *= H

        w0 = xmax - xmin
        h0 = ymax - ymin
        if w0 <= 0 or h0 <= 0:
            continue

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        w_scaled = w0 * scale
        h_scaled = h0 * scale

        x1s = cx - w_scaled / 2.0
        x2s = cx + w_scaled / 2.0
        y1s = cy - h_scaled / 2.0
        y2s = cy + h_scaled / 2.0

        x1 = max(0, min(int(x1s), W))
        x2 = max(0, min(int(x2s), W))
        y1 = max(0, min(int(y1s), H))
        y2 = max(0, min(int(y2s), H))

        if x2 > x1 and y2 > y1:
            net_masks[i, 0, y1:y2, x1:x2] = 1.0

    return net_masks

@torch.no_grad()
def inpaint_with_lama_masks(normalized_batch, masks, lama_model, device='cuda'):
    B2, C2, H2, W2 = normalized_batch.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, C2, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, C2, 1, 1)

    images = normalized_batch * std + mean  # (B2, C2, H2, W2)

    images_padded = pad_to_modulo(images, 8)  # (B2, C2, H_pad, W_pad)
    masks_padded  = pad_to_modulo(masks,  8)  # (B2, 1,  H_pad, W_pad)

    batch = {'image': images_padded, 'mask': masks_padded}
    out   = lama_model(batch)
    inpainted = out['inpainted'].clamp(0, 1)  # (B2, C2, H_pad, W_pad)

    inpainted_cropped = inpainted[:, :, :H2, :W2]
    normalized_out = (inpainted_cropped - mean) / std
    return normalized_out  # (B2, C2, H2, W2)

@torch.no_grad()
def inpaint_with_lama_masks_split2(normalized_batch, masks, lama_model, device='cuda'):
    B, C, H, W = normalized_batch.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, C, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, C, 1, 1)

    images = normalized_batch.to(device) * std + mean
    masks  = masks.to(device)

    images_p = pad_to_modulo(images, 8)
    masks_p  = pad_to_modulo(masks,  8)
    _, _, H_pad, W_pad = images_p.shape

    mid = B // 2
    outs = []
    for s, e in [(0, mid), (mid, B)]:
        batch = {'image': images_p[s:e], 'mask': masks_p[s:e]}
        out = lama_model(batch)['inpainted'].clamp(0, 1)
        out = out[:, :, :H, :W]
        outs.append((out - mean) / std)

    return torch.cat(outs, dim=0)