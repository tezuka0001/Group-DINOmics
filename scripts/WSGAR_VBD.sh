#!/bin/bash

gpu_id=0

common_params=(--supervised --dataset "volleyball" --data_path "./Dataset" --backbone_learnable --backbone_learnable_layers 2 --num_frame 10 --num_total_frame 10 --num_activities 8 --lr 5e-5)

other_params=(--device "$gpu_id" --batch 8 --image_width 512 --image_height 288 --backbone "dinov3" --random_seed 1)

detector_params=(--detector)

epochs=(--epochs 30)

# mask_params=(--ball_mask)
# mask_params=(--ball_lama)

# loss_weight=(--w_ball 10)

# the weights of the model trained with the Person Flow Estimaiton Loss and Object Location Loss
finetune_params=(--pretrained_weights './weights_path.pth')

# Group-relevant Object Location Estimation Loss
pattern1=(--ball_pred --spatial_loss --temporal_loss)

# Person Flow Estimation Loss
pattern2=(--flow_pred --spatial_flow_loss --temporal_flow_loss --spatial_mlp_flow)

# finetune WSGAR ver.
echo "degug1"
CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${finetune_params[@]}" "${detector_params[@]}"
echo "degug2"

# full scratch WSGAR ver.
# echo "degug5"
# CUDA_VISIBLE_DEVICES=$gpu_id python train_test_flow_ball_GAR.py "${common_params[@]}" "${other_params[@]}" "${mask_params[@]}" "${detector_params[@]}" "${loss_weight[@]}" "${pattern1[@]}" "${pattern2[@]}"
# echo "degug6"