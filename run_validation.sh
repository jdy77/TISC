#!/bin/bash

# unet_all_mol_learnable_query Validation Example Script

CUDA_VISIBLE_DEVICES=1 python validate_slice.py \
    --fold 0 \
    --checkpoint /path/to/best/checkpoint.pth \
    --output_dir validation_results/fold0 \
    --normalize minmax \
    --num_workers 4 \
    --seed 42 \
    --model unet_all_mol_learnable_query \
