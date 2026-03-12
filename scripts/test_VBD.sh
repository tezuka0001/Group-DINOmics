#!/bin/bash

gpu_id=0

# common_params=(--dataset "volleyball" --data_path "./Dataset/" --num_frame 10 --num_total_frame 10 --num_activities 8)
common_params=(--dataset "volleyball" --data_path "/home-local/tezuka/" --backbone_learnable --backbone_learnable_layers 2 --num_frame 10 --num_total_frame 10 --num_activities 8 --lr 5e-5)

other_params=(--device "$gpu_id" --batch 8 --image_width 512 --image_height 288 --backbone "dinov3" --random_seed 1)

# detector_params=(--detector)

# mask_params=(--ball_mask)
# mask_params=(--ball_lama)

# the weights of the model trained with the Person Flow Estimaiton Loss and Object Location Loss
# model_path=(--model_path "./weights.pth")
model_path=(--model_path "/home/tezuka/foundation/dinov2_with_attention_extraction/finetune_result/[volleyball]_dino_flow_ball_finetune<2025-08-29_14-43-49>/epoch26_82.65.pth")

CUDA_VISIBLE_DEVICES=$gpu_id python test_flow_ball.py "${common_params[@]}" "${other_params[@]}" "${model_path[@]}" "${mask_params[@]}"