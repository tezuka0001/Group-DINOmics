#!/bin/bash

# GPU番号
gpu_id=0

# 共通パラメータ（配列で定義）
common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 1e-5)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 5e-5)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --backbone_learnable --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 1e-7 --max_lr 1e-6 --weight_decay 1e-6)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --backbone_learnable --num_frame 15 --num_total_frame 15 --num_activities 7 --lr 5e-7 --max_lr 5e-6 --weight_decay 5e-6)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 5e-6 --max_lr 5e-5 --weight_decay 5e-5)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --linear_probing --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 5e-6 --max_lr 5e-5 --weight_decay 5e-5)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --use_lora --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 1e-7 --max_lr 1e-6 --weight_decay 1e-6)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --backbone_full_learnable --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 3e-6 --max_lr 3e-5 --weight_decay 5e-5)

# ViT Blocksを追加した場合の共通パラメータ
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --ViT_Blocks 1 --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 5e-6 --max_lr 5e-5 --weight_decay 5e-5)
# common_params=(--dataset "jrdb" --data_path "/home-local/tezuka/" --backbone_learnable --ViT_Blocks 1 --num_frame 1 --num_total_frame 15 --num_activities 7 --lr 5e-6 --max_lr 5e-5 --weight_decay 5e-5)

# その他のオプション（引数名が正しいか確認する）

# other_params=(--device "$gpu_id" --batch 4 --image_width 896 --image_height 504 --epochs 30 --backbone "dinov2")
#other_params=(--device "$gpu_id" --batch 8 --image_width 448 --image_height 252 --epochs 30 --backbone "dinov2")
# other_params=(--device "$gpu_id" --batch 8 --test_batch 8 --image_width 56 --image_height 56 --epochs 1 --test_freq 1 --backbone "dinov2" --random_seed 1)
# other_params=(--device "$gpu_id" --batch 8 --test_batch 8 --image_width 448 --image_height 252 --epochs 30 --backbone "dinov2" --random_seed 1)
# other_params=(--device "$gpu_id" --batch 8 --test_batch 8 --image_width 476 --image_height 476 --epochs 30 --backbone "dinov2" --random_seed 1)
# other_params=(--device "$gpu_id" --batch 8 --test_batch 8 --image_width 882 --image_height 112 --epochs 30 --backbone "dinov2" --random_seed 1)
other_params=(--device "$gpu_id" --batch 8 --test_batch 8 --image_width 1008 --image_height 128 --epochs 30 --backbone "dinov3" --random_seed 1)
# other_params=(--device "$gpu_id" --batch 32 --test_batch 8 --image_width 1008 --image_height 128 --epochs 30 --backbone "dinov3" --random_seed 1)

# patch 14
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --backbone "dinov2" --random_seed 1)
#以下はclip_lama環境で実行
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --backbone "clip" --random_seed 1)
# patch 16
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --backbone "ViT" --random_seed 1)
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --backbone "MAE" --random_seed 1)

# loss wightの設定
# loss_weight=(--w_inpaint_people 5.0)
# loss_weight=(--w_flow 0.1)
# loss_weight=(--w_ball 10)
# loss_weight=(--w_ball 10 --w_inpaint_people 10.0)

# detectorの設定
# detector_params=(--detector)

# epochのオプション
epochs=(--epochs 30  --flow_gap 2)
# epochs=(--epochs 50)
# epochs=(--epochs 50 --lr_step 5 --lr_step_down 45 --test_freq 2)

# マスクの設定
#mask_params=(--ball_mask --random_mask --ball_inpaint)
#mask_params=(--ball_inpaint)

# mask_params=(--ball_mask)
# mask_params=(--random_mask)
# mask_params=(--ball_mask --random_mask)
# mask_params=(--ball_lama)
#mask_params=(--ball_lama --random_mask)

# mask_params=(--ball_mask --frame_random --inpaint_prob 0.5)
#mask_params=(--ball_mask --batch_random --inpaint_prob 0.5)
# mask_params=(--ball_lama --frame_random --inpaint_prob 0.5)
#mask_params=(--ball_lama --batch_random --inpaint_prob 0.5)

# mask_params=(--people_lama --people_mask_scale 1.2 --inpaint_people_prob 0.1)
# mask_params=(--ball_lama --people_lama --people_mask_scale 1.2 --inpaint_people_prob 0.1)
# mask_params=(--ball_lama --frame_random --inpaint_prob 0.5 --people_lama --people_mask_scale 1.2 --inpaint_people_prob 0.3)

# mask_params=(--people_mask --people_mask_scale 1.2 --inpaint_people_prob 0.5)
# mask_params=(--ball_lama --people_mask --people_mask_scale 1.2 --inpaint_people_prob 0.3)
# mask_params=(--ball_lama --frame_random --inpaint_prob 0.5 --people_mask --people_mask_scale 1.2 --inpaint_people_prob 0.3)

#mask_params=(--ball_lama --lama_model_dir '/home/tezuka/foundation/dinov2_with_attention_extraction/lama/big-lama/')

# 設定パターン1
#pattern1=(--ball_pred --spatial_loss --temporal_loss --future_mask)
#pattern1=(--ball_pred --spatial_loss --temporal_loss --future_mask --test_time_mask)
#pattern1=(--ball_pred --spatial_loss)
#pattern1=(--ball_pred --temporal_loss)
#pattern1=(--ball_pred --future_mask --test_time_mask)
#pattern1=(--ball_pred --future_mask)
pattern1=(--ball_pred --spatial_loss --temporal_loss)
#pattern1=(--ball_pred --temporal_loss --future_mask --test_time_mask)
#pattern1=(--ball_pred --spatial_loss --future_mask --test_time_mask)

# 設定パターン2（必要に応じて切り替え）
# pattern2=(--flow_pred --spatial_flow_loss --temporal_flow_loss --spatial_mlp_flow)
pattern2=(--flow_pred --spatial_flow_loss --temporal_flow_loss --spatial_mlp_flow --center_flow)
# pattern2=(--flow_pred --spatial_flow_loss)
# pattern2=(--flow_pred --temporal_flow_loss --center_flow)

# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${loss_weight[@]}" "${pattern2[@]}"
# echo "degug3"

# CLS, patch token学習
# cls or patch patchの設定，convを使わない場合，pattern1 or pattern2を使う
path_params=(--cls_path)

pattern3=(--person_recon)

# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_cls_patch_for_jrdb.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${path_params[@]}" "${loss_weight[@]}" "${pattern3[@]}"
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_cls_patch_for_jrdb.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${path_params[@]}" "${loss_weight[@]}" "${pattern2[@]}" "${pattern3[@]}" 
# 上を使う
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_cls_patch_for_jrdb.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${loss_weight[@]}" "${pattern3[@]}"
# echo "degug2"

# finetune_params=(--pretrained_weights '/home/tezuka/foundation/dinov2_with_attention_extraction/retrieval_result/[jrdb]_dino_flow_ball_numpy_cls_patch<2026-01-24_19-59-20>/epoch24_0.43.pth')

echo "degug2"
CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_cls_patch_for_jrdb.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${path_params[@]}" "${loss_weight[@]}" "${pattern2[@]}" "${finetune_params[@]}"
echo "degug2"