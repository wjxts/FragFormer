#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python scripts/finetune_fragformer.py --model_path ../models/pretrained/base_0.3/base_50_25000.pth --dataset pharm_ames --weight_decay 0 --dropout 0 --lr 3e-5 --warmup --d_model 512 --n_mol_layers 6 --epochs 50 --warmup_epochs 5 --knodes md ecfp torsion maccs  --vocab_size 500  --order 1 