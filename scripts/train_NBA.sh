#!/bin/bash

gpu_id=0

# common_params=(--dataset "nba" --data_path "./Dataset/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 5e-5)
common_params=(--dataset "nba" --data_path "/home-local/tezuka/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 12 --num_total_frame 71 --num_activities 9 --lr 5e-5)

other_params=(--device "$gpu_id" --batch 8 --image_width 512 --image_height 288 --backbone "dinov3" --random_seed 1)

epochs=(--epochs 50)

# mask_params=(--ball_mask)
# mask_params=(--ball_lama)
# mask_params=(--net_lama --net_extend_to_top --net_extend_side "auto")
mask_params=(--ball_lama --net_lama --net_extend_to_top --net_extend_side "auto")

# loss_weight=(--w_ball 10)

# Group-relevant Object Location Estimation Loss
pattern1=(--ball_pred --spatial_loss --temporal_loss)

# Person Flow Estimation Loss
pattern2=(--flow_pred --spatial_flow_loss --temporal_flow_loss --spatial_mlp_flow)

# Group-relevant Object (Net) Location Estimation Loss
pattern3=(--net_pred --spatial_net_loss --temporal_net_loss --w_net 0.5)

# echo "degug1"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${pattern1[@]}"
# echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${pattern2[@]}"
# echo "degug3"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}"
# echo "degug4"

# if you use the Group-relevant Object (Net) Location Estimation Loss, please use train_test_flow_ball_net.py
# Group-relevant Object (Net) Location Estimation Loss
pattern3=(--net_pred --spatial_net_loss --temporal_net_loss --w_net 0.5)

echo "degug3"
CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_net.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}" "${pattern3[@]}" "${detector_params[@]}"
echo "degug4"