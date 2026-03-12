#!/bin/bash

gpu_id=1

# common_params=(--dataset "nba" --data_path "./Dataset/" --num_frame 12 --num_total_frame 71 --num_activities 9)
common_params=(--dataset "nba" --data_path "/home-local/tezuka/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 12 --num_total_frame 71 --num_activities 9)

other_params=(--device "$gpu_id" --batch 8 --image_width 512 --image_height 288 --backbone "dinov3" --random_seed 1)

# detector_params=(--detector)

# mask_params=(--ball_mask)
# mask_params=(--ball_lama)

# the weights of the model trained with the Person Flow Estimaiton Loss and Object Location Loss
# model_path=(--model_path "./weights.pth")
model_path=(--model_path "/home/tezuka/foundation/dinov2_with_attention_extraction/finetune_result/[nba]_dino_flow_ball_finetune<2025-09-26_12-18-36>/epoch24_43.86.pth")

CUDA_VISIBLE_DEVICES=$gpu_id python test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${model_path[@]}" "${mask_params[@]}"