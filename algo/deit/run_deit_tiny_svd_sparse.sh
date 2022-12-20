#!/bin/bash
python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224 \
--resume 'exp_hr/svd/deit_tiny_1e-4_5e-6_100_mix_head_fc_qk_hid_2/checkpoint_best.pth' \
--data-path /root/data/ILSVRC/Data/CLS-LOC \
--lr 1e-5 \
--weight-decay 1e-8 \
--epochs 100 \
--min-lr 1e-5 \
--output_dir exp_hr/lowrank_sparse/deit_tiny/deit_tiny_info50 \
--mask_path 'masks/deit_tiny_lowrank_hid_2/info_0.5.npy' \
--svd_type 'mix_head_fc_qk' \
--restart_finetune

python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224 \
--resume 'exp_hr/svd/deit_tiny_1e-4_5e-6_100_mix_head_fc_qk_hid_2/checkpoint_best.pth' \
--data-path /root/data/ILSVRC/Data/CLS-LOC \
--lr 1e-5 \
--weight-decay 1e-8 \
--epochs 100 \
--min-lr 1e-5 \
--output_dir exp_hr/lowrank_sparse/deit_tiny/deit_tiny_info80 \
--mask_path 'masks/deit_tiny_lowrank_hid_2/info_0.8.npy' \
--svd_type 'mix_head_fc_qk' \
--restart_finetune

python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224 \
--resume 'exp_hr/svd/deit_tiny_1e-4_5e-6_100_mix_head_fc_qk_hid_2/checkpoint_best.pth' \
--data-path /root/data/ILSVRC/Data/CLS-LOC \
--lr 1e-5 \
--weight-decay 1e-8 \
--epochs 100 \
--min-lr 1e-5 \
--output_dir exp_hr/lowrank_sparse/deit_tiny/deit_tiny_info90 \
--mask_path 'masks/deit_tiny_lowrank_hid_2/info_0.9.npy' \
--svd_type 'mix_head_fc_qk' \
--restart_finetune

python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224 \
--resume 'exp_hr/svd/deit_tiny_1e-4_5e-6_100_mix_head_fc_qk_hid_2/checkpoint_best.pth' \
--data-path /root/data/ILSVRC/Data/CLS-LOC \
--lr 1e-5 \
--weight-decay 1e-8 \
--epochs 100 \
--min-lr 1e-5 \
--output_dir exp_hr/lowrank_sparse/deit_tiny/deit_tiny_info95 \
--mask_path 'masks/deit_tiny_lowrank_hid_2/info_0.95.npy' \
--svd_type 'mix_head_fc_qk' \
--restart_finetune
