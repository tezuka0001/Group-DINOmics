#!/bin/bash

# GPU番号
gpu_id=7
# 今は，flow_numpy_sub_med

# 共通パラメータ（配列で定義）
# common_params=(--supervised --dataset "nba" --data_path "/home-local/tezuka/" --backbone_learnable --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 2e-6 --max_lr 2e-5 --weight_decay 2e-5)
# common_params=(--supervised --dataset "nba" --data_path "/home-local/tezuka/" --backbone_learnable --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 2e-6 --max_lr 2e-5 --weight_decay 2e-5)
# common_params=(--supervised --dataset "nba" --data_path "/home-local/tezuka/" --backbone_learnable --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 1e-7 --max_lr 1e-6 --weight_decay 2e-5)
#common_params=(--dataset "nba" --data_path "/home-local/tezuka/" --dino_learnable --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 1e-6 --max_lr 1e-5 --weight_decay 1e-5)

# cos scheduler
# common_params=(--supervised --dataset "nba" --data_path "/home-local/tezuka/" --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 2e-5)
# common_params=(--supervised --dataset "nba" --data_path "/home-local/tezuka/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 1e-5)
common_params=(--supervised --dataset "nba" --data_path "/home-local/tezuka/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 5e-5)

# その他のオプション（引数名が正しいか確認する）
# other_params=(--device "$gpu_id" --batch 8 --image_width 1024 --image_height 576 --epochs 30 --backbone "dinov3" --random_seed 1)
# other_params=(--device "$gpu_id" --batch 8 --image_width 896 --image_height 504 --epochs 30 --backbone "dinov2")
# other_params=(--device "$gpu_id" --batch 8 --image_width 448 --image_height 252 --epochs 30 --backbone "dinov2")
# other_params=(--device "$gpu_id" --batch 8 --image_width 448 --image_height 252 --epochs 30 --backbone "dinov2" --random_seed 1)
other_params=(--device "$gpu_id" --batch 8 --image_width 512 --image_height 288 --backbone "dinov3" --random_seed 1)

# patch 14
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --epochs 30 --backbone "dinov2" --random_seed 1)
#以下はclip_lama環境で実行
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --epochs 30 --backbone "clip" --random_seed 1)
# patch 16
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --epochs 30 --backbone "ViT" --random_seed 1)
# other_params=(--device "$gpu_id" --batch 8 --image_width 224 --image_height 224 --epochs 30 --backbone "MAE" --random_seed 1)

# compress_params=(--trans_comp)

# loss wightの設定
#loss_weight=(--w_flow 0.1)
# loss_weight=(--w_ball 10)
# loss_weight=(--w_ball 10 --w_sup 0.1)
# loss_weight=(--w_ball 10 --w_inpaint_people 1.0)

# finetuneの設定
# finetune_params=(--pretrained_weights '/home/tezuka/foundation/dinov2_with_attention_extraction/retrieval_result/[volleyball]_dino_flow_ball_numpy<2025-06-11_10-45-18>/epoch22_76.44.pth')
# クラス分類のMLPのみをfinetuneする場合
# finetune_params=(--fix_model --pretrained_weights '/home/tezuka/foundation/dinov2_with_attention_extraction/retrieval_result/[volleyball]_dino_flow_ball_numpy<2025-06-11_10-45-18>/epoch22_76.44.pth')
# 504x896でflow, ball lossで学習した重みw/o ball lama
# finetune_params=(--pretrained_weights '/home/tezuka/foundation/dinov2_with_attention_extraction/retrieval_result/[nba]_dino_flow_ball_numpy<2025-07-25_15-31-34>/epoch45_41.99.pth')
# 504x896でflow, ball lossで学習した重みw/o ball lama
# finetune_params=(--pretrained_weights '/home/tezuka/foundation/dinov2_with_attention_extraction/retrieval_result/[nba]_dino_flow_ball_numpy<2025-07-25_16-14-58>/epoch45_47.35.pth')

# 288x8でflow, ball lossで学習した重みw/o ball lama
finetune_params=(--pretrained_weights "/home/tezuka/foundation/dinov2_with_attention_extraction/finetune_result/[nba]_dino_flow_ball_finetune<2025-09-26_12-18-36>/epoch24_43.86.pth")

# detectorの設定
# detector_params=(--detector)

# epochのオプション
epochs=(--epochs 30)
#epochs=(--epochs 50 --lr_step 15 --lr_step_down 35)

# マスクの設定
#mask_params=(--ball_mask --random_mask --ball_inpaint)
#mask_params=(--ball_inpaint)

#mask_params=(--ball_mask)
#mask_params=(--random_mask)
#mask_params=(--ball_mask --random_mask)
# mask_params=(--ball_lama)
#mask_params=(--ball_lama --random_mask)

#mask_params=(--ball_mask --frame_random --inpaint_prob 0.5)
#mask_params=(--ball_mask --batch_random --inpaint_prob 0.5)
#mask_params=(--ball_lama --frame_random --inpaint_prob 0.5)
#mask_params=(--ball_lama --batch_random --inpaint_prob 0.5)

# mask_params=(--people_lama --people_mask_scale 1.2 --inpaint_people_prob 0.3)
# mask_params=(--ball_lama --people_lama --people_mask_scale 1.2 --inpaint_people_prob 0.1)
# mask_params=(--ball_lama --frame_random --inpaint_prob 0.5 --people_lama --people_mask_scale 1.2 --inpaint_people_prob 0.3)

# mask_params=(--people_mask --people_mask_scale 1.2 --inpaint_people_prob 0.3)
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
pattern2=(--flow_pred --spatial_flow_loss --temporal_flow_loss)
# pattern2=(--flow_pred --spatial_flow_loss)
# pattern2=(--flow_pred --temporal_flow_loss)

# finetune ver.
echo "degug2"
CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${finetune_params[@]}"
echo "degug3"

# full scratch ver.
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}"
# echo "degug3"
# full scratch trans_comp ver.
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${compress_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}"
# echo "degug3"

# detector_free_params=(--detector_free)
# full scratch WSGAR detector-free ver.
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${detector_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}" "${detector_free_params[@]}"
# echo "degug3"

# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${pattern1[@]}" "${pattern2[@]}"
# echo "degug1"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}"
# echo "degug1"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${pattern1[@]}" "${pattern2[@]}"
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}"
# echo "degug3"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${pattern1[@]}"
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${pattern1[@]}"
# echo "degug3"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${pattern2[@]}"
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${pattern2[@]}"
# echo "degug3"

#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${detector_params[@]}" "${pattern1[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${detector_params[@]}" "${pattern1[@]}" "${pattern2[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}"


#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${pattern2[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${pattern1[@]}" "${pattern2[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${pattern1[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${pattern1[@]}" "${pattern2[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${detector_params[@]}" "${pattern1[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${detector_params[@]}" "${pattern2[@]}"
#CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${detector_params[@]}" "${pattern1[@]}" "${pattern2[@]}"