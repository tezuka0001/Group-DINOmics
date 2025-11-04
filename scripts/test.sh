#!/bin/bash

gpu_id=0

common_params=(--dataset "volleyball" --data_path "./Dataset/" --num_frame 10 --num_total_frame 10 --num_activities 8)
# common_params=(--dataset "nba" --data_path "./Dataset/" --num_frame 12 --num_total_frame 71 --num_activities 9)

other_params=(--device "$gpu_id" --batch 8 --image_width 512 --image_height 288 --backbone "dinov3" --random_seed 1)

# detector_params=(--detector)

# mask_params=(--ball_mask)
# mask_params=(--ball_lama)

model_path=(--model_path "./weights.pth")

CUDA_VISIBLE_DEVICES=$gpu_id python test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${model_path[@]}" "${mask_params[@]}"