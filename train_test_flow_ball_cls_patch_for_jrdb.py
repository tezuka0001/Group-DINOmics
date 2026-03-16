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

# モデル・データローダのインポート（データローダは (images, bbox_gt, activities) を返す前提）
import models.models_flow_ball_cls_patch as models
# import models.models_flow_ball_net as models
from util.utils import *
#from dataloader.dataloader_bbox_flow import read_dataset
from dataloader.dataloader_bbox_flow_detector_jrdb import read_dataset
from sklearn.manifold import TSNE
from tqdm import tqdm

# lamaのimport
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
parser.add_argument('--nheads_agg', default=4, type=int, help='number of heads for partial context aggregation')

parser.add_argument('--recon_loss', default=0.1, type=float, help='reconstruction loss weight')

# GPU
parser.add_argument('--device', default="0, 1", type=str, help='GPU device')

# Load model
# parser.add_argument('--load_model', action='store_true', help='load model')
# parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

parser.add_argument('--head_list', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], type=list, help='select_list')

parser.add_argument('--hidden_size', default=1024, type=int, help='hidden size')

parser.add_argument('--detector', action='store_true', help='use detector')

# backbone parameters
parser.add_argument('--backbone_learnable', action='store_true', help='use learnable last layer')
parser.add_argument('--backbone_full_learnable', action='store_true', help='use full learnable last layer')
parser.add_argument('--backbone_learnable_layers', default=1, type=int, help='number of learnable layers')
parser.add_argument('--backbone', default='dinov2', type=str, help='backbone model, dinov2, clip, ViT, MAE, resnet50, vgg16, vgg19')
parser.add_argument('--linear_probing', action='store_true', help='use linear probing for backbone')
parser.add_argument('--ViT_arch', default='vit-l', type=str, help='vit-l, vit-b')
parser.add_argument('--ViT_Blocks', default=0, type=int, help='number of blocks for ViT backbone')
parser.add_argument('--use_lora', action='store_true', help='use LoRA for backbone')
parser.add_argument('--spatial_mlp_flow', action='store_true', help='use spatial flow mlp for backbone')

parser.add_argument('--ball_mask', action='store_true', help='use ball mask')
parser.add_argument('--random_mask', action='store_true', help='use random mask')
parser.add_argument('--ball_inpaint', action='store_true', help='use ball inpaint')
parser.add_argument('--people_mask', action='store_true', help='use people mask')
parser.add_argument('--ball_pred', action='store_true', help='use ball prediction')
parser.add_argument('--flow_pred', action='store_true', help='use flow prediction')
parser.add_argument('--flow_recon', action='store_true', help='use flow reconstruction')
parser.add_argument('--person_recon', action='store_true', help='use person reconstruction')

# flow_estimation from pacth tokens
parser.add_argument('--flow_patch', action='store_true', help='use person reconstruction')

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
parser.add_argument('--w_inpaint_people', default=0.0, type=float, help='weight of inpaint people loss')

parser.add_argument('--cls_path', action='store_true', help='use cls path')
parser.add_argument('--cls_concat', action='store_true', help='use cls concat')
parser.add_argument('--patch_concat', action='store_true', help='use patch concat')
parser.add_argument('--pooling_method', default='avg', type=str, help='pooling method for patch tokens')
# parser.add_argument('--patch_en', action='store_true', help='use patch encoder')
# parser.add_argument('--patch_pool', action='store_true', help='patch token pooling')
parser.add_argument('--patch_path', action='store_true', help='use patch path')
parser.add_argument('--patch_maxpool', action='store_true', help='patch token maxpooling')
parser.add_argument('--patch_avgpool', action='store_true', help='patch token avgpooling')
parser.add_argument('--patch_cnn', action='store_true', help='patch token cnn')

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

parser.add_argument('--center_flow', action='store_true', help='use center flow patch')
parser.add_argument('--flow_gap', type=int, default=7, help='flow gap between frames')

parser.add_argument('--spatial_backbone_mlp', action='store_true', help='use spatial backbone mlp')
parser.add_argument('--spatial_mlp_ball', action='store_true', help='use spatial mlp for ball for backbone')
parser.add_argument('--temp_mlp_flow', action='store_true', help='use temporal flow mlp for backbone')

parser.add_argument('--net_pred', action='store_true', help='use net prediction')
parser.add_argument('--spatial_net_loss', action='store_true', help='use spatial net loss')
parser.add_argument('--temporal_net_loss', action='store_true', help='use temporal net loss')
parser.add_argument('--net_lama', action='store_true', help='use net inpaint with LaMa')
parser.add_argument('--net_extend_to_top', action='store_true', help='extend net bbox to top')
parser.add_argument('--net_extend_side', default='none', type=str, help='extend net bbox side: none, left, right, both')
parser.add_argument('--inpaint_prob_net', type=float, default=1.0, help='probability of net inpainting')
parser.add_argument('--net_mask_scale', type=float, default=1.0, help='scale for net mask')
parser.add_argument('--w_net', default=1.0, type=float, help='weight of net loss')
parser.add_argument('--crop_flow_loss', action='store_true', help='use crop flow loss')

parser.add_argument('--pretrained_weights', type=str, default='', help='pretrained weights path')

args = parser.parse_args()
args.use_flow = False
args.use_flow_numpy = args.flow_pred or args.flow_recon
# args.use_flow = args.flow_pred or args.flow_recon
# args.use_flow_numpy = False

# グローバル変数（これまでの hit@1 の最高値とそのエポック、および現在のベストモデルのパス）
best_hit_rate = 0.0
best_hit_epoch = 0
best_model_path = None  # 現在のベストモデルのパス

best_val_loss = float('inf')
best_val_epoch = 0
best_val_model_path = None

def save_best_model(epoch, model, optimizer, scheduler, hit_1, save_path):
    global best_model_path
    # これまで通り、例: epoch12_85.50.pth
    result_path = os.path.join(save_path, 'epoch%d_%.2f.pth' % (epoch, hit_1))
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'hit@1': hit_1
    }
    torch.save(state, result_path)
    # 以前のベストモデルが存在していれば削除
    if best_model_path is not None and os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Removed old model: {best_model_path}")
    best_model_path = result_path
    print(f"Saved new best model at epoch {epoch} with hit@1 {hit_1:.2f}% to {result_path}")
    
def save_best_model_by_loss(epoch, model, optimizer, scheduler, val_loss, save_path):
    global best_val_model_path
    result_path = os.path.join(save_path, f'best_val_epoch{epoch:03d}_loss{val_loss:.4f}.pth')
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss': val_loss,
        'tag': 'best_val'
    }
    torch.save(state, result_path)

    if best_val_model_path is not None and os.path.exists(best_val_model_path):
        os.remove(best_val_model_path)
        print(f"Removed old best-val model: {best_val_model_path}")
    best_val_model_path = result_path
    print(f"Saved new BEST-VAL model at epoch {epoch} (val_loss={val_loss:.4f}) -> {result_path}")

@torch.no_grad()
def test_evaluation(train_loader_for_val, test_loader, model, args, save_path):
    """
    ベストモデルを用いてテストデータの評価を行い、混同行列（CSV, JPG）と
    hit@1～hit@5, precision@1～precision@5 を計算し、その結果を標準出力および
    save_path 配下の test_log.txt に保存します。また、テストデータの特徴量空間を
    t-SNE で2次元可視化し、JPG で保存します。
    """
    model.eval()
    # --- 学習データから特徴量空間を作成 ---
    train_features_list = []
    train_labels_list = []
    for sample in tqdm(train_loader_for_val, desc="Extracting Train Features for Test"):
        if args.use_flow or args.use_flow_numpy:
            images, activities, _, _, bboxes, _ = sample
        else:
            images, activities, _, bboxes, _ = sample
        images = images.cuda()
        input_data = {'images': images, 'bboxes': bboxes}
        ret_dic = model(input_data)
        video_features = ret_dic['video_features']
        train_features_list.append(video_features.cpu().numpy())
        train_labels_list.append(activities[:, 0].cpu().numpy())
    train_features = np.vstack(train_features_list)
    train_labels = np.hstack(train_labels_list)

    # --- テストデータから特徴量空間を作成 ---
    test_features_list = []
    test_labels_list = []
    for sample in tqdm(test_loader, desc="Extracting Test Features"):
        if args.use_flow or args.use_flow_numpy:
            images, activities, ball_gt, flow, bboxes, _ = sample
        else:
            images, activities, ball_gt, bboxes, _ = sample
        images = images.cuda()
        input_data = {'images': images, 'bboxes': bboxes}
        ret_dic = model(input_data)
        video_features = ret_dic['video_features']
        test_features_list.append(video_features.cpu().numpy())
        test_labels_list.append(activities[:, 0].cpu().numpy())
    test_features = np.vstack(test_features_list)
    test_labels = np.hstack(test_labels_list)
    
    # --- k=1～5 による hit と precision の計算 ---
    k_list = [1, 2, 3, 4, 5]
    nbrs_k = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(train_features)
    distances_k, indices_k = nbrs_k.kneighbors(test_features)

    # --- 混同行列の算出と保存 ---
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
            # 各クエリの上位 k 件における正解数
            correct_count = np.sum(neighbor_labels[:k] == test_labels[i])
            # precision@k は、(正解数 / k)
            precision_sum[k] += correct_count / k
            # hit@k は、正解が1件でもあればカウント
            if correct_count > 0:
                hit_counts[k] += 1

    total_queries = len(test_labels)
    hit_rates = {k: hit_counts[k] / total_queries * 100.0 for k in k_list}
    precisions = {k: precision_sum[k] / total_queries * 100.0 for k in k_list}

    # --- t-SNE による特徴量空間の可視化（学習データのみ） ---
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

    # --- t-SNE による特徴量空間の可視化（テストデータのみ） ---
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

    # --- 結果の出力（標準出力および test_log.txt に保存） ---
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
            
@torch.no_grad()
def test_evaluation_multilabel(train_loader, test_loader, model, args, save_path):
    """
    マルチラベル retrieval task 評価：
      1. 学習・テストデータから動画特徴量と multi-hot ラベルを抽出
      2. k 最近傍を探索
      3. IoU@k を計算（各サンプル上位 k の近傍との最大 IoU の平均）
      4. IoU しきい値ごとの Hit@k（0.3,0.5,0.7,1.0）を計算
      5. マルチラベル分類として Precision/Recall/F1（samples avg）も計算
      6. retrieval_iou.txt, multilabel_confusion_matrix.csv, classification_report.txt/csv,
         test_log.txt を save_path 配下に保存
    """

    # クラス名リストを直接定義（数字ラベルではなく文字列で出力）
    class_names = [
        "commuting",
        "resting",
        "conversing",
        "office working",
        "walking",
        "waiting",
        "sitting"
    ]

    model.eval()

    # --- 特徴量・マルチラベル収集（学習データ） ---
    train_feats, train_labels = [], []
    for sample in tqdm(train_loader, desc="Extracting Train Features"):
        if args.use_flow_numpy:
            images, _, bboxes, _, activities, _, *_ = sample
        else:
            images, bboxes, _, activities, _, *_ = sample
        images = images.cuda()
        bboxes = bboxes.cuda()
        activities = activities.cuda()           # shape: (B,1,C)
        ret = model({'images': images, 'bboxes': bboxes})
        train_feats.append(ret['video_features'].cpu().numpy())  # (B,D)
        train_labels.append(activities[:,0,:].cpu().numpy())    # (B,C)

    train_features = np.vstack(train_feats)   # (N_train,D)
    train_labels   = np.vstack(train_labels)  # (N_train,C)

    # --- 特徴量・マルチラベル収集（テストデータ） ---
    test_feats, test_labels = [], []
    for sample in tqdm(test_loader, desc="Extracting Test Features"):
        if args.use_flow_numpy:
            images, _, bboxes, _, activities, _, *_ = sample
        else:
            images, bboxes, _, activities, _, *_ = sample
        images = images.cuda()
        bboxes = bboxes.cuda()
        activities = activities.cuda()
        ret = model({'images': images, 'bboxes': bboxes})
        test_feats.append(ret['video_features'].cpu().numpy())
        test_labels.append(activities[:,0,:].cpu().numpy())

    test_features = np.vstack(test_feats)   # (N_test,D)
    test_labels   = np.vstack(test_labels)  # (N_test,C)

    # --- k 最近傍探索 & IoU@k, しきい値 Hit@k 計算 ---
    k_list     = [1,2,3,4,5]
    thresholds = [0.3, 0.5, 0.7, 1.0]
    nbrs = NearestNeighbors(n_neighbors=max(k_list), algorithm='brute') \
           .fit(train_features)
    _, indices = nbrs.kneighbors(test_features)  # (N_test, max_k)

    # 初期化
    iou_sums       = {k: 0.0 for k in k_list}
    thr_hit_counts = {thr: {k: 0 for k in k_list} for thr in thresholds}

    # 各サンプルごとに
    for i, neigh_idxs in enumerate(indices):
        y_true = test_labels[i]  # (C,)
        for k in k_list:
            neigh_labels = train_labels[neigh_idxs[:k]]  # (k, C)
            # 最大 IoU を計算
            max_iou = 0.0
            for y_pred in neigh_labels:
                inter = np.logical_and(y_true, y_pred).sum()
                union = np.logical_or(y_true, y_pred).sum()
                iou = inter/union if union>0 else 0.0
                if iou > max_iou:
                    max_iou = iou
            iou_sums[k] += max_iou
            # しきい値ごとのヒット判定
            for thr in thresholds:
                if max_iou >= thr:
                    thr_hit_counts[thr][k] += 1

    n_test = len(test_labels)
    # 平均 IoU@k
    iou_at_k = {k: iou_sums[k]/n_test for k in k_list}
    # しきい値別 Hit@k (%)
    thr_hit_rates = {
        thr: {k: thr_hit_counts[thr][k]/n_test*100.0 for k in k_list}
        for thr in thresholds
    }

    # --- retrieval_iou.txt に保存 ---
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "retrieval_iou.txt"), "w") as f:
        f.write("IoU@k:\n")
        for k in k_list:
            f.write(f"IoU@{k}: {iou_at_k[k]:.4f}\n")
        f.write("\nHit@k for IoU thresholds:\n")
        for thr in thresholds:
            f.write(f"IoU ≥ {thr}:\n")
            for k in k_list:
                f.write(f"  Hit@{k}: {thr_hit_rates[thr][k]:.2f}%\n")
    print(f"Retrieval results saved to: {save_path}/retrieval_iou.txt")

    # --- マルチラベル分類として Precision/Recall/F1 (samples avg) ---
    preds = train_labels[indices[:,0]]  # (N_test,C)
    p_samples = precision_score(test_labels, preds, average='samples', zero_division=0)*100
    r_samples = recall_score   (test_labels, preds, average='samples', zero_division=0)*100
    f_samples = f1_score       (test_labels, preds, average='samples', zero_division=0)*100
    print("\nAs multi-label classification (k=1 neighbor):")
    print(f"  Precision (samples): {p_samples:.2f}%")
    print(f"  Recall    (samples): {r_samples:.2f}%")
    print(f"  F1-score  (samples): {f_samples:.2f}%")

    # --- multilabel_confusion_matrix を CSV に保存 ---
    mcm = multilabel_confusion_matrix(test_labels, preds)
    mcm_csv = os.path.join(save_path, "multilabel_confusion_matrix.csv")
    with open(mcm_csv, "w") as f:
        f.write("Class,TN,FP,FN,TP\n")
        for i, cm in enumerate(mcm):
            label = class_names[i]
            TN, FP = cm[0]
            FN, TP = cm[1]
            f.write(f"{label},{TN},{FP},{FN},{TP}\n")
    print(f"Multilabel confusion matrix saved to: {mcm_csv}")

    # --- classification_report をテキスト & CSV で保存 ---
    report_txt = os.path.join(save_path, "classification_report.txt")
    report_csv = os.path.join(save_path, "classification_report.csv")
    report_str = classification_report(
        test_labels, preds,
        target_names=class_names,
        zero_division=0
    )
    with open(report_txt, "w") as f:
        f.write(report_str)
    report_dict = classification_report(
        test_labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(report_csv)
    print(f"Classification report saved to: {report_txt}, {report_csv}")

    # --- その他平均指標 (micro/macro/weighted) ---
    p_micro    = precision_score(test_labels, preds, average='micro',    zero_division=0)*100
    r_micro    = recall_score   (test_labels, preds, average='micro',    zero_division=0)*100
    f_micro    = f1_score       (test_labels, preds, average='micro',    zero_division=0)*100
    p_macro    = precision_score(test_labels, preds, average='macro',    zero_division=0)*100
    r_macro    = recall_score   (test_labels, preds, average='macro',    zero_division=0)*100
    f_macro    = f1_score       (test_labels, preds, average='macro',    zero_division=0)*100
    p_weighted = precision_score(test_labels, preds, average='weighted', zero_division=0)*100
    r_weighted = recall_score   (test_labels, preds, average='weighted', zero_division=0)*100
    f_weighted = f1_score       (test_labels, preds, average='weighted', zero_division=0)*100

    # --- test_log.txt に主要指標まとめて保存 ---
    lines = [
        f"P (micro):    {p_micro:.2f}%, R (micro):    {r_micro:.2f}%, F1 (micro):    {f_micro:.2f}%",
        f"P (macro):    {p_macro:.2f}%, R (macro):    {r_macro:.2f}%, F1 (macro):    {f_macro:.2f}%",
        f"P (weighted): {p_weighted:.2f}%, R (weighted): {r_weighted:.2f}%, F1 (weighted): {f_weighted:.2f}%",
        f"P (samples):  {p_samples:.2f}%, R (samples):  {r_samples:.2f}%, F1 (samples):  {f_samples:.2f}%",
        f"Saved: retrieval_iou.txt, {mcm_csv}, {report_txt}, {report_csv}"
    ]
    log_path = os.path.join(save_path, "test_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary metrics saved to: {log_path}")

    return {
        "precision_micro":    p_micro,
        "recall_micro":       r_micro,
        "f1_micro":           f_micro,
        "precision_macro":    p_macro,
        "recall_macro":       r_macro,
        "f1_macro":           f_macro,
        "precision_weighted": p_weighted,
        "recall_weighted":    r_weighted,
        "f1_weighted":        f_weighted,
        "precision_samples":  p_samples,
        "recall_samples":     r_samples,
        "f1_samples":         f_samples,
        **{f"IoU@{k}": iou_at_k[k] for k in k_list}
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
            "train_loss_person_recon": train_log['loss_person_recon'],
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
                "val_loss_person_recon": test_log['loss_person_recon'],
                "hit@1": test_log['hit@1'],
                "hit@2": test_log['hit@2'],
                "hit@3": test_log['hit@3'],
                "best_hit_rate": test_log['best_hit_rate'],
                "best_hit_epoch": test_log['best_hit_epoch'],
                "epoch": test_log['epoch'],
            })
        elif args.dataset == 'jrdb':
            wandb.log({
                "val_loss": test_log['loss'],
                'val_loss_flow_spatial': test_log['loss_flow_spatial'],
                'val_loss_flow_temporal': test_log['loss_flow_temporal'],
                "val_loss_person_recon": test_log['loss_person_recon'],
                "val_epoch_time": test_log['time'],
                "iou@1": test_log['IoU@1'],
                "iou@2": test_log['IoU@2'],
                "iou@3": test_log['IoU@3'],
                "best_iou_rate": test_log['best_hit_rate'],
                "best_iou_epoch": test_log['best_hit_epoch'],
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
        if args.pretrained_weights != '':
            project_name = "flow_ball_volleyball_cls_patch_finetune"
        else:
            project_name = "flow_ball_volleyball_numpy_cls_patch"
    elif args.dataset == 'nba':
        if args.pretrained_weights != '':
            project_name = "flow_ball_nba_cls_patch_finetune"
        else:
            project_name = "flow_ball_nba_numpy_cls_patch"
    elif args.dataset == 'jrdb':
        if args.pretrained_weights != '':
            project_name = "flow_ball_jrdb_cls_patch_finetune"
        else:
            project_name = "flow_ball_jrdb_numpy_cls_patch"
    wandb.init(
        project=project_name,
        name=f'{args.dataset}_experiment_{time.strftime("%Y%m%d-%H%M%S")}',
        config=args,
    )

def train(args, train_loader, model, criterion, optimizer, epoch, lama_model):
    """
    1エポック分の学習を行う関数です。
    """
    epoch_timer = Timer()
    losses = AverageMeter()
    losses_bbox = AverageMeter()
    losses_temp_bbox = AverageMeter()
    losses_future_bbox = AverageMeter()
    losses_flow_spatial = AverageMeter()
    losses_flow_temporal = AverageMeter()
    losses_person_recon = AverageMeter()
    grad_norms = AverageMeter()
    model.train()
    
    if args.dataset == 'nba':
        mask_size = 40 * args.image_width // 1280
        # mask_size = 40 * args.image_width // 1280
        mask_size_min, mask_size_max = 40 * args.image_width // 1280, 80 * args.image_width // 1280
    elif args.dataset == 'volleyball':
        #mask_size = 60 * args.image_width // 1280
        mask_size = 40 * args.image_width // 1280
        mask_size_min, mask_size_max = 20 * args.image_width // 1280, 60 * args.image_width // 1280
        
    # 「最悪ケース（バッチ内すべてのフレームをインペイント）」に
    #   対応できる固定長バッファを GPU に確保
    MAX_INPAINT = args.batch * args.num_frame   # = 最大の B*T
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
        loss_person_recon = torch.tensor(0.0).cuda()
        
        # if args.use_flow or args.use_flow_numpy:
        #         images, activities, ball_gt, flow, bboxes, _ = sample
        #         flow = flow.cuda()
        # else:
        #     images, activities, ball_gt, bboxes, _ = sample
        #     activities = activities.cuda()
        #     images = images.cuda()
        #     ball_gt = ball_gt.cuda()
        #     bboxes = bboxes.cuda()
        # elif args.dataset == 'jrdb':
        #     if args.use_flow_numpy:
        #         images, flow, bboxes, _, activities, bboxes_num, *_ = sample
        #         flow = flow.cuda()
        #     else:                
        #         images, bboxes, _, activities, bboxes_num, *_ = sample
        #     images = images.cuda()
        #     bboxes = bboxes.cuda()
        #     bboxes_num = bboxes_num.cuda()
        #     activities = activities.cuda()
        
        if args.dataset in ['volleyball', 'nba']:
            if args.use_flow or args.use_flow_numpy:
                images, activities, _, _, bboxes, _ = sample
            else:
                images, activities, _, bboxes, _ = sample

        elif args.dataset == 'jrdb':
            if args.use_flow_numpy:
                images, flow, bboxes, _, activities, bboxes_num, *_ = sample
                flow = flow.cuda()
            else:
                images, bboxes, _, activities, bboxes_num, *_ = sample
            bboxes_num = bboxes_num.cuda()
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        images = images.cuda()
        bboxes = bboxes.cuda()
        activities = activities.cuda()
            
        B, T, C, H, W = images.shape
        if args.dataset == 'volleyball':
            N = 12
        elif args.dataset == 'nba':
            N = 10
        elif args.dataset == 'jrdb':
            N = 60
        # if args.ball_mask:
        #     images = apply_ball_mask(images, ball_gt, mask_size=mask_size)
        if args.ball_mask:
            B, T = images.shape[:2]  # 現バッチのサイズ

            # (1) ボール座標を画素単位に変換
            ball_gt_real = torch.zeros(B, T, 2, device='cuda', dtype=torch.long)
            ball_gt_real[..., 0] = (ball_gt[..., 0] * args.image_width ).long()
            ball_gt_real[..., 1] = (ball_gt[..., 1] * args.image_height).long()

            # (2) (B*T, …) の一次元列に整形
            imgs_r   = images.view(B * T, 3, args.image_height, args.image_width)
            coords_r = ball_gt_real.view(B * T, 2)

            # (3) インペイント対象のインデックスを決定
            if args.frame_random:
                mask = torch.rand(B * T, device='cuda') < args.inpaint_prob
            elif args.batch_random:
                batch_mask = [random.random() < args.inpaint_prob for _ in range(B)]
                mask = torch.zeros(B * T, dtype=torch.bool, device='cuda')
                for b in range(B):
                    if batch_mask[b]:
                        mask[b * T : (b + 1) * T] = True
            else:
                # フラグがどちらも False の場合：すべてのフレームをインペイント
                mask = torch.ones(B * T, dtype=torch.bool, device='cuda')

            # (4) マスクされたフレーム数を確認し、LaMa を実行
            num = int(mask.sum().item())
            if num > 0:
                # 対象フレームを事前バッファにコピー
                buf_imgs[:num].copy_(imgs_r[mask])
                buf_coords[:num].copy_(coords_r[mask])
                out = apply_ball_mask_prob(buf_imgs[:num], buf_coords[:num], mask_size=mask_size)

                # 結果を元画像に書き戻す
                imgs_r[mask] = out

            # (5) 5次元に戻す
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
            elif args.dataset == 'jrdb':
                ori_h, ori_w = 480, 3760
            bboxes_rescaled = bboxes_flat.clone()
            bboxes_rescaled[..., 0] = (bboxes_flat[..., 0] / ori_h) * H
            bboxes_rescaled[..., 2] = (bboxes_flat[..., 2] / ori_h) * H
            bboxes_rescaled[..., 1] = (bboxes_flat[..., 1] / ori_w) * W
            bboxes_rescaled[..., 3] = (bboxes_flat[..., 3] / ori_w) * W
            # scale を指定できる。例：args.people_scale = 1.2 など
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
            # (1) フラット化
            imgs_flat   = images.view(B * T, C, H, W)   # (B⋅T, C, H, W)
            ball_flat   = ball_gt.view(B * T, 2)   # (B⋅T, 2)
            bboxes_flat = bboxes.view(B * T, N, 4)      # (B⋅T, N, 4)

            # ----------------------------
            # (2) ボール用マスクを作成
            # ----------------------------
            if args.ball_lama:
                ball_flat_real = torch.zeros(B * T, 2, device='cuda', dtype=torch.long)
                ball_flat_real[..., 0] = (ball_flat[..., 0] * args.image_width ).long()
                ball_flat_real[..., 1] = (ball_flat[..., 1] * args.image_height).long()
                # フレームごと or バッチごとのどちらかで True/False を作成
                if args.frame_random:
                    # フレーム単位で確率判定
                    do_ball = (torch.rand(B * T, device='cuda') < args.inpaint_prob)
                elif args.batch_random:
                    # バッチ単位で確率判定 → True のバッチすべてのフレームをマスク
                    batch_mask = [random.random() < args.inpaint_prob for _ in range(B)]
                    do_ball = torch.zeros(B * T, dtype=torch.bool, device='cuda')
                    for b in range(B):
                        if batch_mask[b]:
                            do_ball[b * T : (b + 1) * T] = True
                else:
                    do_ball = torch.ones(B * T, dtype=torch.bool, device='cuda')
                # まず全フレームで「座標から円形 or 正方形マスク」を作る
                ball_masks = generate_ball_masks(
                    normalized_batch=imgs_flat,
                    centers=ball_flat_real,
                    mask_size=mask_size,
                    mask_shape=args.mask_shape,
                    device='cuda'
                )  # → (B*T, 1, H, W)
                # do_ball==False のフレームはゼロマスクにする
                ball_masks[~do_ball] = 0.0
            else:
                ball_masks = torch.zeros(B * T, 1, H, W, device='cuda')

            # ----------------------------
            # (3) 人物用マスクを作成（関数呼び出し）
            # ----------------------------
            if args.people_lama:
                if args.dataset == 'volleyball' or args.dataset == 'nba':
                    ori_h, ori_w = 720, 1280
                elif args.dataset == 'jrdb':
                    ori_h, ori_w = 480, 3760
                bboxes_rescaled = bboxes_flat.clone()
                bboxes_rescaled[..., 0] = (bboxes_flat[..., 0] / ori_h) * H
                bboxes_rescaled[..., 2] = (bboxes_flat[..., 2] / ori_h) * H
                bboxes_rescaled[..., 1] = (bboxes_flat[..., 1] / ori_w) * W
                bboxes_rescaled[..., 3] = (bboxes_flat[..., 3] / ori_w) * W
                # scale を指定できる。例：args.people_scale = 1.2 など
                people_masks, people_masks_flags = generate_people_masks(
                    normalized_batch=imgs_flat,
                    bboxes=bboxes_rescaled,
                    scale=args.people_mask_scale,
                    inpaint_prob=args.inpaint_people_prob,
                    device='cuda'
                )  # (B⋅T, 1, H, W)
            else:
                people_masks = torch.zeros(B * T, 1, H, W, device='cuda')

            # ----------------------------
            # (4) ボールマスク＋人物マスクを OR 合成
            # ----------------------------
            combined_masks = torch.clamp(ball_masks + people_masks, 0.0, 1.0)  # (B⋅T, 1, H, W)

            # ----------------------------
            # (5) 一度に LaMa に投げてインペイント
            # ----------------------------
            inpainted_flat = inpaint_with_lama_masks(
                normalized_batch=imgs_flat,
                masks=combined_masks,
                lama_model=lama_model,
                device='cuda'
            )  # (B⋅T, C, H, W)

            # (6) 元の (B, T, C, H, W) に戻す
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
        
        if args.flow_pred:
            if args.dataset in ['volleyball', 'nba']:
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
                    # loss_flow_spatial = criterion(spatial_flow_pred, flow_gt)
                    loss_flow_spatial = weighted_flow_loss_mse(spatial_flow_pred, flow_gt, masked_flags, args.w_inpaint_people)
                    loss += args.w_flow * loss_flow_spatial
                if args.temporal_flow_loss:
                    # loss_flow_temporal = criterion(temp_flow_pred, flow_gt)
                    loss_flow_temporal = weighted_flow_loss_mse(temp_flow_pred, flow_gt, masked_flags, args.w_inpaint_people)
                    loss += args.w_flow * loss_flow_temporal
                
            elif args. dataset == 'jrdb':
                '''
                # --- ① パディングされたボックスを除外するマスク ---
                # bboxes: (B, T, N, 4)
                existence_mask = (bboxes.abs().sum(dim=-1) != 0)  # shape: [B, T, N]

                # --- ② 重心座標の計算 ---
                bboxes_x_center = ((bboxes[..., 0] + bboxes[..., 2]) // 2).long()
                bboxes_y_center = ((bboxes[..., 1] + bboxes[..., 3]) // 2).long()
                batch_idx       = torch.arange(B, device='cuda').view(B, 1, 1).expand(B, T, N)
                frame_idx       = torch.arange(T, device='cuda').view(1, T, 1).expand(B, T, N)

                # --- ③ flow の正規化／ダウンサンプリング（numpy フローを使う場合のみ） ---
                if args.use_flow_numpy:
                    breakpoint()
                    flow = normalize_flow_minmax(flow)
                    flow = flow.permute(0, 1, 4, 2, 3)  # (B, T, 2, H, W)
                    scale_x = args.image_width  / 3760
                    scale_y = args.image_height /  480
                    factor  = 14 if args.backbone in ['dinov2', 'clip'] else 16
                    bboxes_x_center = ((bboxes_x_center * scale_x) // factor).long()
                    bboxes_y_center = ((bboxes_y_center * scale_y) // factor).long()

                # --- ④ GT フローの抽出 ---
                flow_x = flow[batch_idx, frame_idx, 0, bboxes_y_center, bboxes_x_center]
                flow_y = flow[batch_idx, frame_idx, 1, bboxes_y_center, bboxes_x_center]
                flow_gt = torch.stack([flow_x, flow_y], dim=-1)  # (B, T, N, 2)

                # --- ⑤ モデル予測の取り出し ---
                spatial_flow_pred = ret_dic['pred_flow_spatial']  # (B, T, N, 2)
                temp_flow_pred    = ret_dic['pred_flow_temporal'] # (B, T, N, 2)
                '''
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if 'people_masks_flags' in locals():
                    people_mask_flags = people_masks_flags.view(B, T, N).bool()
                else:
                    people_mask_flags = torch.zeros((B, T, N), dtype=torch.bool, device=device)

                existence_mask = (
                    torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
                    < bboxes_num.unsqueeze(-1)
                )  # True: 実在人物
                # final_mask = people_mask_flags | (~existence_mask)  # True: マスク対象

                bboxes_x_center = ((bboxes[..., 0] + bboxes[..., 2]) // 2).long()
                bboxes_y_center = ((bboxes[..., 1] + bboxes[..., 3]) // 2).long()
                batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, N)
                frame_idx = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)

                if args.use_flow_numpy:
                    flow = normalize_flow_minmax(flow)
                    scale_x = args.image_width / 3760
                    scale_y = args.image_height / 480
                    factor = 14 if args.backbone in ['dinov2', 'clip'] else 16
                    bboxes_x_center = ((bboxes_x_center * scale_x) // factor).long()
                    bboxes_y_center = ((bboxes_y_center * scale_y) // factor).long()

                flow_x = flow[batch_idx, frame_idx, 0, bboxes_y_center, bboxes_x_center]
                flow_y = flow[batch_idx, frame_idx, 1, bboxes_y_center, bboxes_x_center]
                flow_gt = torch.stack([flow_x, flow_y], dim=-1)

                spatial_flow_pred = ret_dic['pred_flow_spatial']
                temp_flow_pred = ret_dic['pred_flow_temporal']

                # --- ⑥ Zero‑padded を無視する MSE 損失の計算 ---
                if args.spatial_flow_loss:
                    # loss_flow_spatial = masked_mse_loss(
                    #     spatial_flow_pred, flow_gt, existence_mask
                    # )
                    loss_flow_spatial = masked_mse_loss_center(
                        spatial_flow_pred, flow_gt, existence_mask
                    )
                    # loss_flow_spatial = masked_weighted_mse_loss(
                    #     spatial_flow_pred, flow_gt, existence_mask
                    # )
                    # loss_flow_spatial = masked_weighted_mse_loss_batch_norm(
                    #     spatial_flow_pred, flow_gt, existence_mask
                    # )
                    # loss_flow_spatial = masked_weighted_mse_loss_frame_norm(
                    #     spatial_flow_pred, flow_gt, existence_mask
                    # )
                    loss += args.w_flow * loss_flow_spatial

                if args.temporal_flow_loss:
                    # loss_flow_temporal = masked_mse_loss(
                    #     temp_flow_pred, flow_gt, existence_mask
                    # )
                    loss_flow_temporal = masked_mse_loss_center(
                        temp_flow_pred, flow_gt, existence_mask
                    )
                    # loss_flow_temporal = masked_weighted_mse_loss(
                    #     temp_flow_pred, flow_gt, existence_mask
                    # )
                    # loss_flow_temporal = masked_weighted_mse_loss_batch_norm(
                    #     temp_flow_pred, flow_gt, existence_mask
                    # )
                    # loss_flow_temporal = masked_weighted_mse_loss_frame_norm(
                    #     temp_flow_pred, flow_gt, existence_mask
                    # )
                    loss += args.w_flow * loss_flow_temporal
                    
        if args.person_recon:
            pred_person_recon = ret_dic['pred_person_recon']
            patch_tokens = ret_dic['patch_tokens']
            pooled = pool_from_patch_tokens_mean(
                patch_tokens_bt=patch_tokens,
                bboxes_xyxy=bboxes,
                image_size=(args.image_height, args.image_width),
                empty_fallback='nearest',
            )
            loss_person_recon = criterion(pred_person_recon, pooled)
            loss += loss_person_recon
        
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

        
        if loss.item() == 0:
            loss = loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            #nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        else:
            # クリッピングしない場合は自前でノルムを計算
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm_sq ** 0.5
        grad_norms.update(grad_norm)
        optimizer.step()
        
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
        elif args.flow_pred or args.person_recon:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)
            losses_person_recon.update(loss_person_recon.item(), B)

    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'loss_bbox': losses_bbox.avg,
        'loss_temp_bbox': losses_temp_bbox.avg,
        'loss_future_bbox': losses_future_bbox.avg,
        'loss_flow_spatial': losses_flow_spatial.avg,
        'loss_flow_temporal': losses_flow_temporal.avg,
        'loss_person_recon': losses_person_recon.avg,
        'grad_norm': grad_norms.avg,
    }
    log_metrics_to_wandb(train_log, test_log=None)
    return train_log

'''
@torch.no_grad()
def validate(train_loader_for_val, test_loader, model, criterion, epoch, optimizer, scheduler, save_path):
    """
    学習データから特徴量空間を作成し、テストデータに対して hit@1,2,3 を計算する評価用関数です。
    """
    global best_hit_rate, best_hit_epoch
    epoch_timer = Timer()
    losses = AverageMeter()
    losses_bbox = AverageMeter()
    losses_temp_bbox = AverageMeter()
    losses_future_bbox = AverageMeter()
    losses_flow_spatial = AverageMeter()
    losses_flow_temporal = AverageMeter()
    model.eval()

    train_features_list = []
    train_labels_list = []
    for sample in tqdm(train_loader_for_val, desc="Extracting Train Features"):
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            if args.use_flow or args.use_flow_numpy:
                images, activities, _, _, bboxes, _ = sample
            else:
                images, activities, _, bboxes, _ = sample
        elif args.dataset == 'jrdb':
            if args.use_flow_numpy:
                images, _, bboxes, _, activities, bboxes_num, *_ = sample
            else:                
                images, bboxes, _, activities, bboxes_num, *_ = sample
            activities = activities.cuda()
        images = images.cuda()
        bboxes = bboxes.cuda()
        activities = activities.cuda()
        input_data = {'images': images, 'bboxes': bboxes}
        ret_dic = model(input_data)
        video_features = ret_dic['video_features']
        train_features_list.append(video_features.cpu().numpy())
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            train_labels_list.append(activities[:, 0].cpu().numpy())
        elif args.dataset == 'jrdb':
            train_labels_list.append(activities[:, 0, :].cpu().numpy())
    train_features = np.vstack(train_features_list)
    print("train_features NaN:", np.isnan(train_features).any())
    if args.dataset == 'volleyball' or args.dataset == 'nba':
        train_labels = np.hstack(train_labels_list)
    elif args.dataset == 'jrdb':
        train_labels = np.vstack(train_labels_list)

    query_features_list = []
    query_labels_list = []
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
            images = images.cuda()
            ball_gt = ball_gt.cuda()
            bboxes = bboxes.cuda()
        elif args.dataset == 'jrdb':
            if args.use_flow_numpy:
                images, flow, bboxes, _, activities, bboxes_num, *_ = sample
                flow = flow.cuda()
            else:                
                images, bboxes, _, activities, bboxes_num, *_ = sample
            images = images.cuda()
            bboxes = bboxes.cuda()
            bboxes_num = bboxes_num.cuda()
            activities = activities.cuda()
        B, T, _, _, _ = images.shape
        if args.dataset == 'volleyball':
            N = 12
        elif args.dataset == 'nba':
            N = 10
        elif args.dataset == 'jrdb':
            N = 60
        input_data = {'images': images, 'bboxes': bboxes}
        ret_dic = model(input_data)
        
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
                    flow = normalize_flow_minmax(flow)
                    scale_x = args.image_width / 1280
                    scale_y = args.image_height / 720
                    if args.backbone == 'dinov2' or args.backbone == 'clip':
                        bboxes_x_center = ((bboxes_x_center * scale_x) // 14).long()
                        bboxes_y_center = ((bboxes_y_center * scale_y) // 14).long()
                    elif args.backbone == 'dinov3' or args.backbone == 'ViT' or args.backbone == 'MAE':
                        bboxes_x_center = ((bboxes_x_center * scale_x) // 16).long()
                        bboxes_y_center = ((bboxes_y_center * scale_y) // 16).long()
                flow_x = flow[batch_indices, frame_indices, 0, bboxes_y_center, bboxes_x_center]
                flow_y = flow[batch_indices, frame_indices, 1, bboxes_y_center, bboxes_x_center]
                flow_gt = torch.stack([flow_x, flow_y], dim=-1)
                spatial_flow_pred = ret_dic['pred_flow_spatial']
                temp_flow_pred = ret_dic['pred_flow_temporal']
                if args.spatial_flow_loss:
                    # loss_flow_spatial = criterion(spatial_flow_pred, flow_gt)
                    loss_flow_spatial = weighted_flow_loss_mse(spatial_flow_pred, flow_gt, masked_flags, args.w_inpaint_people)
                    loss += args.w_flow * loss_flow_spatial
                if args.temporal_flow_loss:
                    # loss_flow_temporal = criterion(temp_flow_pred, flow_gt)
                    loss_flow_temporal = weighted_flow_loss_mse(temp_flow_pred, flow_gt, masked_flags, args.w_inpaint_people)
                    loss += args.w_flow * loss_flow_temporal
                    
            elif args.dataset == 'jrdb':
                    # --- 前処理はそのまま ---
                if 'people_masks_flags' in locals():
                    people_mask_flags = people_masks_flags.view(B, T, N).bool()
                else:
                    people_mask_flags = torch.zeros((B, T, N), dtype=torch.bool, device='cuda')

                # bboxes_num: Tensor of shape [B, T], 各フレームに検出された人物数
                # N = 60 (最大人数／パディング後の人数)
                # existence_mask: True が「実際に検出された人物分」
                existence_mask = (
                    torch.arange(N, device=bboxes.device)
                        .view(1, 1, N)
                        .expand(B, T, N)
                        < bboxes_num.unsqueeze(-1)
                )  # shape: [B, T, N], dtype=bool

                # 最終的なマスク：人物マスク or パディング分を True に
                final_mask = people_mask_flags | (~existence_mask)

                # 重心位置の計算はそのまま
                bboxes_x_center = ((bboxes[..., 0] + bboxes[..., 2]) // 2).long()
                bboxes_y_center = ((bboxes[..., 1] + bboxes[..., 3]) // 2).long()
                batch_idx = torch.arange(B, device='cuda').view(B, 1, 1).expand(B, T, N)
                frame_idx = torch.arange(T, device='cuda').view(1, T, 1).expand(B, T, N)

                if args.use_flow_numpy:
                    flow = normalize_flow_minmax(flow)
                    scale_x = args.image_width / 3760
                    scale_y = args.image_height / 480
                    if args.backbone in ['dinov2', 'clip']:
                        factor = 14
                    else:  # ViT or MAE
                        factor = 16
                    bboxes_x_center = ((bboxes_x_center * scale_x) // factor).long()
                    bboxes_y_center = ((bboxes_y_center * scale_y) // factor).long()

                flow_x = flow[batch_idx, frame_idx, 0, bboxes_y_center, bboxes_x_center]
                flow_y = flow[batch_idx, frame_idx, 1, bboxes_y_center, bboxes_x_center]
                flow_gt = torch.stack([flow_x, flow_y], dim=-1)

                spatial_flow_pred = ret_dic['pred_flow_spatial']
                temp_flow_pred    = ret_dic['pred_flow_temporal']

                if args.spatial_flow_loss:
                    loss_flow_spatial = weighted_flow_loss_mse(
                        spatial_flow_pred, flow_gt, final_mask, args.w_inpaint_people
                    )
                    loss += args.w_flow * loss_flow_spatial

                if args.temporal_flow_loss:
                    loss_flow_temporal = weighted_flow_loss_mse(
                        temp_flow_pred, flow_gt, final_mask, args.w_inpaint_people
                    )
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
        
        query_features_list.append(video_features_test.cpu().numpy())
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            query_labels_list.append(activities[:, 0].cpu().numpy())
        elif args.dataset == 'jrdb':
            query_labels_list.append(activities[:, 0, :].cpu().numpy())
    query_features = np.vstack(query_features_list)
    print("query_features NaN:", np.isnan(query_features).any())
    if args.dataset == 'volleyball' or args.dataset == 'nba':
        query_labels = np.hstack(query_labels_list)
    elif args.dataset == 'jrdb':
        query_labels = np.vstack(query_labels_list)
    
    if args.dataset == 'volleyball' or args.dataset == 'nba':
        k_list = [1, 2, 3]
        hit_rates, precisions = calculate_hit_and_precision(
            train_features=train_features,
            train_labels=train_labels,
            query_features=query_features,
            query_labels=query_labels,
            k_list=k_list
        )

        hit_rate_1 = hit_rates[1]
        hit_rate_2 = hit_rates[2]
        hit_rate_3 = hit_rates[3]

        if hit_rate_1 > best_hit_rate:
            best_hit_rate = hit_rate_1
            best_hit_epoch = epoch
            save_best_model(epoch, model, optimizer, scheduler, hit_rate_1, save_path)

        val_log = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            'loss': losses.avg,
            'loss_bbox': losses_bbox.avg,
            'loss_temp_bbox': losses_temp_bbox.avg,
            'loss_future_bbox': losses_future_bbox.avg,
            'loss_flow_spatial': losses_flow_spatial.avg,
            'loss_flow_temporal': losses_flow_temporal.avg,
            'hit@1': hit_rate_1,
            'hit@2': hit_rate_2,
            'hit@3': hit_rate_3,
            'best_hit_rate': best_hit_rate,
            'best_hit_epoch': best_hit_epoch,
        }
    
    elif args.dataset == 'jrdb':
        # --- k 最近傍探索 ---
        k_list = [1, 2, 3]
        max_k = max(k_list)
        nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='brute').fit(train_features)
        _, indices = nbrs.kneighbors(query_features)  # indices: (N_test, max_k)

        # --- IoU@k の計算 ---
        iou_sums = {k: 0.0 for k in k_list}
        for i, neigh_idxs in enumerate(indices):
            y_true = query_labels[i]  # (C,)
            for k in k_list:
                neigh_labels = train_labels[neigh_idxs[:k]]  # (k, C)
                # 各近傍との IoU を計算し、最大値を取る
                ious = []
                for y_pred in neigh_labels:
                    inter = np.logical_and(y_true, y_pred).sum()
                    union = np.logical_or(y_true, y_pred).sum()
                    ious.append(inter / union if union > 0 else 0.0)
                iou_sums[k] += max(ious)

        n_test = len(query_labels)
        iou_at_k = {k: iou_sums[k] / n_test for k in k_list}
        
        iou_at_1 = iou_at_k[1]
        iou_at_2 = iou_at_k[2]
        iou_at_3 = iou_at_k[3]
        
        if iou_at_1 > best_hit_rate:
            best_hit_rate = iou_at_1
            best_hit_epoch = epoch
            save_best_model(epoch, model, optimizer, scheduler, iou_at_1, save_path)
            
        val_log = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            'loss': losses.avg,
            'loss_flow_spatial': losses_flow_spatial.avg,
            'loss_flow_temporal': losses_flow_temporal.avg,
            'IoU@1': iou_at_1,
            'IoU@2': iou_at_2,
            'IoU@3': iou_at_3,
            'best_hit_rate': best_hit_rate,
            'best_hit_epoch': best_hit_epoch,
        }
    
    log_metrics_to_wandb(train_log=None, test_log=val_log)
    return val_log
'''

@torch.no_grad()
def validate(train_loader_for_val, test_loader, model, criterion, epoch, optimizer, scheduler, save_path):
    """
    学習データから特徴量空間を作成し、テストデータに対して hit@1,2,3（volleyball/nba）、
    もしくは IoU@1,2,3（jrdb）を計算する評価用関数です。
    全て GPU 上で保持・計算します（距離計算は query chunk で分割）。
    """
    global best_hit_rate, best_hit_epoch, best_val_loss, best_val_epoch
    epoch_timer = Timer()
    losses = AverageMeter()
    losses_bbox = AverageMeter()
    losses_temp_bbox = AverageMeter()
    losses_future_bbox = AverageMeter()
    losses_flow_spatial = AverageMeter()
    losses_flow_temporal = AverageMeter()
    losses_person_recon = AverageMeter()
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================
    # Train features/labels を GPU 上に保存
    # =========================
    train_features_list = []
    train_labels_list = []

    for sample in tqdm(train_loader_for_val, desc="Extracting Train Features"):
        if args.dataset in ['volleyball', 'nba']:
            if args.use_flow or args.use_flow_numpy:
                images, activities, _, _, bboxes, _ = sample
            else:
                images, activities, _, bboxes, _ = sample

        elif args.dataset == 'jrdb':
            if args.use_flow_numpy:
                images, _, bboxes, _, activities, bboxes_num, *_ = sample
            else:
                images, bboxes, _, activities, bboxes_num, *_ = sample
            bboxes_num = bboxes_num.to(device, non_blocking=True)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        images = images.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        activities = activities.to(device, non_blocking=True)

        input_data = {'images': images, 'bboxes': bboxes}
        ret_dic = model(input_data)
        video_features = ret_dic['video_features']  # [B, D] on GPU

        train_features_list.append(video_features.detach())

        if args.dataset in ['volleyball', 'nba']:
            train_labels_list.append(activities[:, 0].detach())         # [B]
        else:  # jrdb
            train_labels_list.append(activities[:, 0, :].detach())      # [B, C] (multi-hot)

    train_features = torch.cat(train_features_list, dim=0)  # [N_train, D] on GPU
    print("train_features NaN:", torch.isnan(train_features).any().item())

    train_labels = torch.cat(train_labels_list, dim=0)      # [N_train] or [N_train, C] on GPU

    # =========================
    # Test features/labels を GPU 上に保存 & 損失計算
    # =========================
    query_features_list = []
    query_labels_list = []

    for sample in tqdm(test_loader, desc="Extracting Test Features"):
        loss = torch.tensor(0.0, device=device)
        loss_bbox = torch.tensor(0.0, device=device)
        loss_temp_bbox = torch.tensor(0.0, device=device)
        loss_future_bbox = torch.tensor(0.0, device=device)
        loss_flow_spatial = torch.tensor(0.0, device=device)
        loss_flow_temporal = torch.tensor(0.0, device=device)

        if args.dataset in ['volleyball', 'nba']:
            if args.use_flow or args.use_flow_numpy:
                images, activities, ball_gt, flow, bboxes, _ = sample
                flow = flow.to(device, non_blocking=True)
            else:
                images, activities, ball_gt, bboxes, _ = sample

            images = images.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            ball_gt = ball_gt.to(device, non_blocking=True)
            activities = activities.to(device, non_blocking=True)

        elif args.dataset == 'jrdb':
            if args.use_flow_numpy:
                images, flow, bboxes, _, activities, bboxes_num, *_ = sample
                flow = flow.to(device, non_blocking=True)
            else:
                images, bboxes, _, activities, bboxes_num, *_ = sample

            images = images.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            bboxes_num = bboxes_num.to(device, non_blocking=True)
            activities = activities.to(device, non_blocking=True)

        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        B, T, _, _, _ = images.shape
        if args.dataset == 'volleyball':
            N = 12
        elif args.dataset == 'nba':
            N = 10
        elif args.dataset == 'jrdb':
            N = 60
        else:
            N = bboxes.shape[2]

        input_data = {'images': images, 'bboxes': bboxes}
        ret_dic = model(input_data)

        # ===== Flow loss =====
        if args.flow_pred:
            if args.dataset in ['volleyball', 'nba']:
                if 'people_masks_flags' in locals():
                    masked_flags = people_masks_flags.view(B, T, N).bool()
                else:
                    masked_flags = torch.zeros((B, T, N), dtype=torch.bool, device=device)

                bboxes_x_center = (((bboxes[:, :, :, 0] + bboxes[:, :, :, 2]) // 2).long())
                bboxes_y_center = (((bboxes[:, :, :, 1] + bboxes[:, :, :, 3]) // 2).long())

                batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, N)
                frame_indices = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)

                if args.use_flow_numpy:
                    flow = normalize_flow_minmax(flow)
                    scale_x = args.image_width / 1280
                    scale_y = args.image_height / 720
                    if args.backbone == 'dinov2' or args.backbone == 'clip':
                        bboxes_x_center = ((bboxes_x_center * scale_x) // 14).long()
                        bboxes_y_center = ((bboxes_y_center * scale_y) // 14).long()
                    elif args.backbone in ['dinov3', 'ViT', 'MAE', 'dino', 'franca', 'siglip', 'siglip2'] or (args.backbone == 'clip' and args.ViT_arch in ["vit-b", "vit-l"]):
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

            elif args.dataset == 'jrdb':
                if 'people_masks_flags' in locals():
                    people_mask_flags = people_masks_flags.view(B, T, N).bool()
                else:
                    people_mask_flags = torch.zeros((B, T, N), dtype=torch.bool, device=device)

                existence_mask = (
                    torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
                    < bboxes_num.unsqueeze(-1)
                )  # True: 実在人物
                # final_mask = people_mask_flags | (~existence_mask)  # True: マスク対象

                bboxes_x_center = ((bboxes[..., 0] + bboxes[..., 2]) // 2).long()
                bboxes_y_center = ((bboxes[..., 1] + bboxes[..., 3]) // 2).long()
                batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, N)
                frame_idx = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)

                if args.use_flow_numpy:
                    flow = normalize_flow_minmax(flow)
                    scale_x = args.image_width / 3760
                    scale_y = args.image_height / 480
                    factor = 14 if args.backbone in ['dinov2', 'clip'] else 16
                    bboxes_x_center = ((bboxes_x_center * scale_x) // factor).long()
                    bboxes_y_center = ((bboxes_y_center * scale_y) // factor).long()

                flow_x = flow[batch_idx, frame_idx, 0, bboxes_y_center, bboxes_x_center]
                flow_y = flow[batch_idx, frame_idx, 1, bboxes_y_center, bboxes_x_center]
                flow_gt = torch.stack([flow_x, flow_y], dim=-1)

                spatial_flow_pred = ret_dic['pred_flow_spatial']
                temp_flow_pred = ret_dic['pred_flow_temporal']

                if args.spatial_flow_loss:
                    # loss_flow_spatial = weighted_flow_loss_mse(spatial_flow_pred, flow_gt, final_mask, args.w_inpaint_people)
                    # loss_flow_saptial = masked_mse_loss(spatial_flow_pred, flow_gt, existence_mask)
                    loss_flow_spatial = masked_mse_loss_center(spatial_flow_pred, flow_gt, existence_mask)
                    # loss_flow_spatial = masked_weighted_flow_loss_mse(spatial_flow_pred, flow_gt, final_mask, args.w_inpaint_people)
                    loss += args.w_flow * loss_flow_spatial
                if args.temporal_flow_loss:
                    # loss_flow_temporal = weighted_flow_loss_mse(temp_flow_pred, flow_gt, final_mask, args.w_inpaint_people)
                    # loss_flow_temporal = masked_mse_loss(temp_flow_pred, flow_gt, existence_mask)
                    loss_flow_temporal = masked_mse_loss_center(temp_flow_pred, flow_gt, existence_mask)
                    # loss_flow_temporal = masked_weighted_flow_loss_mse(temp_flow_pred, flow_gt, final_mask, args.w_inpaint_people)
                    loss += args.w_flow * loss_flow_temporal
        # ===== Person reconstruction loss（jrdb 前提）=====
        if args.person_recon:
            pred_person_recon = ret_dic['pred_person_recon']
            patch_tokens = ret_dic['patch_tokens']
            pooled = pool_from_patch_tokens_mean(
                patch_tokens_bt=patch_tokens,
                bboxes_xyxy=bboxes,
                image_size=(args.image_height, args.image_width),
                empty_fallback='nearest',
            )
            loss_person_recon = criterion(pred_person_recon, pooled)
            loss += loss_person_recon
            
        # ===== Ball loss（volleyball/nba 前提。jrdb で ball_gt が無いなら事故るのでガード）=====
        if args.ball_pred and args.dataset in ['volleyball', 'nba']:
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
                    future_bbox_pred = future_bbox_pred.view(B, T, T, -1).permute(1, 0, 2, 3)  # [T,B,T,4]
                    future_bbox_pred = future_bbox_pred.reshape(T, B * T, -1)                  # [T,B*T,4]
                    future_bbox_pred_valid = future_bbox_pred[:, valid_frame_mask_flat]
                    ball_gt_valid_expand = ball_gt_valid.expand_as(future_bbox_pred_valid)
                    loss_future_bbox = criterion(future_bbox_pred_valid, ball_gt_valid_expand)
                    loss += args.w_ball * loss_future_bbox

        video_features_test = ret_dic['video_features']  # [B, D] on GPU

        # ===== 損失の集計 =====
        if args.ball_pred and args.flow_pred and args.dataset in ['volleyball', 'nba']:
            losses.update(loss.item(), B)
            losses_flow_spatial.update(loss_flow_spatial.item(), B)
            losses_flow_temporal.update(loss_flow_temporal.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
        elif args.ball_pred and args.dataset in ['volleyball', 'nba']:
            losses.update(loss.item(), B)
            losses_bbox.update(loss_bbox.item(), B)
            losses_temp_bbox.update(loss_temp_bbox.item(), B)
            losses_future_bbox.update(loss_future_bbox.item(), B)
        else:
            losses.update(loss.item(), B)
            if args.flow_pred:
                losses_flow_spatial.update(loss_flow_spatial.item(), B)
                losses_flow_temporal.update(loss_flow_temporal.item(), B)
            if args.person_recon:
                losses_person_recon.update(loss_person_recon.item(), B)

        # ===== 特徴とラベルを GPU 上で保存 =====
        query_features_list.append(video_features_test.detach())
        if args.dataset in ['volleyball', 'nba']:
            query_labels_list.append(activities[:, 0].detach())      # [B]
        else:  # jrdb
            query_labels_list.append(activities[:, 0, :].detach())   # [B,C]

    query_features = torch.cat(query_features_list, dim=0)  # [N_test, D] on GPU
    print("query_features NaN:", torch.isnan(query_features).any().item())
    query_labels = torch.cat(query_labels_list, dim=0)      # [N_test] or [N_test,C] on GPU

    # =========================
    # Retrieval metrics on GPU
    # =========================
    k_list = [1, 2, 3]
    max_k = max(k_list)

    # 距離行列 [N_test, N_train]（小さいほど近い）
    match_score = torch.cdist(query_features, train_features, p=2)
    match_score_argsort = torch.argsort(match_score, dim=1)  # 昇順

    if args.dataset in ['volleyball', 'nba']:
        num_query = query_labels.shape[0]

        hit_rates = {}
        precisions = {}

        for k in k_list:
            topk_idx = match_score_argsort[:, :k]      # [N_test, k]
            topk_labels = train_labels[topk_idx]       # [N_test, k]

            q = query_labels.view(-1, 1).expand(num_query, k)  # [N_test, k]
            correct = (topk_labels == q)                        # [N_test, k]

            hit_rate_k = (correct.any(dim=1).float().mean()).item() * 100.0
            precision_k = ((correct.float().sum(dim=1) / k).mean()).item() * 100.0

            hit_rates[k] = hit_rate_k
            precisions[k] = precision_k

        hit_rate_1, hit_rate_2, hit_rate_3 = hit_rates[1], hit_rates[2], hit_rates[3]

        if hit_rate_1 > best_hit_rate:
            best_hit_rate = hit_rate_1
            best_hit_epoch = epoch
            save_best_model(epoch, model, optimizer, scheduler, hit_rate_1, save_path)

        global best_val_loss, best_val_epoch
        if losses.avg < best_val_loss:
            best_val_loss = losses.avg
            best_val_epoch = epoch
            save_best_model_by_loss(epoch, model, optimizer, scheduler, best_val_loss, save_path)

        val_log = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            'loss': losses.avg,
            'loss_bbox': losses_bbox.avg,
            'loss_temp_bbox': losses_temp_bbox.avg,
            'loss_future_bbox': losses_future_bbox.avg,
            'loss_flow_spatial': losses_flow_spatial.avg,
            'loss_flow_temporal': losses_flow_temporal.avg,
            'loss_person_recon': losses_person_recon.avg,
            'hit@1': hit_rate_1,
            'hit@2': hit_rate_2,
            'hit@3': hit_rate_3,
            'best_hit_rate': best_hit_rate,
            'best_hit_epoch': best_hit_epoch,
        }

    elif args.dataset == 'jrdb':
        # query_labels/train_labels: [N, C] の multi-hot 想定
        train_bin = (train_labels > 0).bool()
        query_bin = (query_labels > 0).bool()

        topk_idx = match_score_argsort[:, :max_k]   # [N_test, max_k]
        neigh = train_bin[topk_idx]                 # [N_test, max_k, C]

        inter = (neigh & query_bin.unsqueeze(1)).sum(dim=-1).float()
        union = (neigh | query_bin.unsqueeze(1)).sum(dim=-1).float().clamp_min(1.0)
        iou = inter / union                         # [N_test, max_k]

        iou_at_k = {}
        for k in k_list:
            best_iou = iou[:, :k].max(dim=1).values
            iou_at_k[k] = best_iou.mean().item()

        iou_at_1, iou_at_2, iou_at_3 = iou_at_k[1], iou_at_k[2], iou_at_k[3]

        if iou_at_1 > best_hit_rate:
            best_hit_rate = iou_at_1
            best_hit_epoch = epoch
            save_best_model(epoch, model, optimizer, scheduler, iou_at_1, save_path)
            
        # global best_val_loss, best_val_epoch
        if losses.avg < best_val_loss:
            best_val_loss = losses.avg
            best_val_epoch = epoch
            save_best_model_by_loss(epoch, model, optimizer, scheduler, best_val_loss, save_path)

        val_log = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            'loss': losses.avg,
            'loss_flow_spatial': losses_flow_spatial.avg,
            'loss_flow_temporal': losses_flow_temporal.avg,
            'loss_person_recon': losses_person_recon.avg,
            'IoU@1': iou_at_1,
            'IoU@2': iou_at_2,
            'IoU@3': iou_at_3,
            'best_hit_rate': best_hit_rate,
            'best_hit_epoch': best_hit_epoch,
        }

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    log_metrics_to_wandb(train_log=None, test_log=val_log)
    return val_log

def worker_init_fn(worker_id):
    seed = args.random_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)

def main():
    global args, best_hit_rate, best_hit_epoch, best_model_path

    # initialize_wandb(args)

    # time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # exp_name = '[%s]_dino_flow_ball_numpy_cls_patch<%s>' % (args.dataset, time_str)
    # save_path = './retrieval_result/%s' % exp_name
    # os.makedirs(save_path, exist_ok=True)
    
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if args.pretrained_weights != '':
        exp_name = '[%s]_dino_flow_ball_cls_patch_finetune<%s>' % (args.dataset, time_str)
        save_path = './finetune_result/%s' % exp_name
    else:
        exp_name = '[%s]_dino_flow_ball_numpy_cls_patch<%s>' % (args.dataset, time_str)
        save_path = './retrieval_result/%s' % exp_name
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
    elif args.dataset == 'jrdb':
        train_set, train_set_for_val, test_set = read_dataset(args)
    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    train_loader_for_val = data.DataLoader(train_set_for_val, batch_size=args.feature_batch, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    model = models.Ball_detect_model(args)
    model = torch.nn.DataParallel(model).cuda()
    
    if args.ball_lama or args.people_lama:
        lama_model = load_lama_model(args.lama_model_dir, checkpoint_name=args.checkpoint_name, device='cuda')
    else:
        lama_model = None

    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    criterion = nn.MSELoss().cuda()
    # criterion = RelativeMSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=args.max_lr, 
                                                #    step_size_up=args.lr_step, step_size_down=args.lr_step_down, 
                                                #    mode='triangular2', cycle_momentum=False)
    # 一般的
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,            # 30epochで1サイクル
        eta_min=1e-6         # 最小学習率（好みで0〜1e-6程度）
    )                                         

    start_epoch = 1
    
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- Train at epoch #%d' % epoch)
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, lama_model)
        print_log(save_path, 'Train - Epoch: %d, Loss: %.4f, Time: %.1f sec' % (epoch, train_log['loss'], train_log['time']))
        current_lr = scheduler.get_last_lr()[0]
        print('Current learning rate: %f' % current_lr)
        wandb.log({"learning_rate": current_lr, "epoch": epoch})
        scheduler.step()

        if epoch % args.test_freq == 0:
            print_log(save_path, '----- Validate at epoch #%d' % epoch)
            val_log = validate(train_loader_for_val, test_loader, model, criterion, epoch, optimizer, scheduler, save_path)
            if args.dataset == 'volleyball' or args.dataset == 'nba':
                print_log(save_path, 'Validation - Epoch: %d, Loss: %.4f, Time: %.1f sec, Hit@1: %.2f%%, Hit@2: %.2f%%, Hit@3: %.2f%%' %
                        (epoch, val_log['loss'], val_log['time'], val_log['hit@1'], val_log['hit@2'], val_log['hit@3']))
                print_log(save_path, 'Best Hit@1: %.2f%% at epoch #%d.' % (val_log['best_hit_rate'], val_log['best_hit_epoch']))
            elif args.dataset == 'jrdb':
                print_log(save_path, 'Validation - Epoch: %d, Loss: %.4f, Time: %.1f sec, IoU@1: %.2f%%, IoU@2: %.2f%%, IoU@3: %.2f%%' %
                        (epoch, val_log['loss'], val_log['time'], val_log['IoU@1'], val_log['IoU@2'], val_log['IoU@3']))
                print_log(save_path, 'Best IoU@1: %.2f%% at epoch #%d.' % (val_log['best_hit_rate'], val_log['best_hit_epoch']))

    if best_model_path is not None:
        print("Training complete. Loading best model from:", best_model_path)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            test_evaluation(train_loader_for_val, test_loader, model, args, save_path)
        elif args.dataset == 'jrdb':
            test_evaluation_multilabel(train_loader_for_val, test_loader, model, args, save_path)
    else:
        print("No best model was saved.")
        
    if best_val_model_path is not None:
        print("Loading best-val model:", best_val_model_path)
        checkpoint = torch.load(best_val_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        eval_dir_val = os.path.join(save_path, "eval_best_val")
        os.makedirs(eval_dir_val, exist_ok=True)
        if args.dataset == 'volleyball' or args.dataset == 'nba':
            test_evaluation(train_loader_for_val, test_loader, model, args, eval_dir_val)
        elif args.dataset == 'jrdb':
            test_evaluation_multilabel(train_loader_for_val, test_loader, model, args, eval_dir_val)
    else:
        print("No best val model was saved.")

if __name__ == '__main__':
    main()
