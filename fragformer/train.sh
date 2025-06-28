#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port 12102 scripts/train_fragformer.py --save_path ../models/pretrained/base_0.3 --n_threads 8 --n_devices 2 --n_steps 25000 --mask_rate 0.3 --d_model 512 --n_mol_layers 6 --attn_drop 0.1 --feat_drop 0.1 --batch_size 4096 --knodes md ecfp torsion maccs --vocab_size 500  --order 1 