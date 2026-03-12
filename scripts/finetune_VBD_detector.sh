#!/bin/bash

gpu_id=0

common_params=(--dataset "volleyball" --data_path "./Dataset/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 10 --num_total_frame 10 --num_activities 8 --lr 3.0e-5)

other_params=(--device "$gpu_id" --batch 8 --image_width 512 --image_height 288 --backbone "dinov3" --random_seed 1)

detector_params=(--detector)

epochs=(--epochs 30)

# mask_params=(--ball_mask)
mask_params=(--ball_lama)
# mask_params=(--net_lama --net_mask_scale 1.3)
# mask_params=(--ball_lama --net_lama --net_mask_scale 1.3)

# loss_weight=(--w_ball 10)

# the weights of the model trained with the Person Flow Estimaiton Loss
finetune_params=(--pretrained_weights './flow_weights_path.pth')

# Group-relevant Object (Ball) Location Estimation Loss
pattern1=(--ball_pred --spatial_loss --temporal_loss)

# Person Flow Estimation Loss
pattern2=(--flow_pred --spatial_flow_loss --temporal_flow_loss --spatial_mlp_flow)

# Group-relevant Object (Net) Location Estimation Loss
pattern3=(--net_pred --spatial_net_loss --temporal_net_loss)

echo "degug1"
CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${finetune_params[@]}" "${pattern1[@]}" "${detector_params[@]}"
echo "degug2"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${finetune_params[@]}" "${pattern2[@]}" "${detector_params[@]}"
# echo "degug3"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${epochs[@]}" "${mask_params[@]}" "${loss_weight[@]}" "${finetune_params[@]}" "${pattern1[@]}" "${pattern2[@]}" "${detector_params[@]}"
# echo "degug4"