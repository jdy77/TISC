#!/usr/bin/env bash

# unet_all_mol_learnable_query Train Example Script

python train_slice.py \
    --model unet_all_mol_learnable_query \
    --fold 0 \
    --splits_json /path/to/splits_final.json \
    --epochs 100 \
    --warm_up 10 \
    --point_loss_weight 20 \
    --coarse_weight 0.8 \
    --exp_name fold2/exp_name \
    --log_loss \
    --cldice \
    --main_slice_weight 2.0 \
    --elastic  --rotation --zoom \
    --save_epochly \
    --learnable_query_ckpt /path/to/learnable_query/fold0/checkpoints/best.pth \
