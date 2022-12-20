# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Functions for saving average attention map
"""
import os

import torch
import numpy as np

import utils

def init(model, args):
    for tmp_i in range(12):
        model.blocks[tmp_i].attn.num_attention.fill_(0)
        model.blocks[tmp_i].attn.attention_sum.fill_(0)


def save(model, args):
    # Stack.
    attn_weights = []
    attn_num = model.blocks[0].attn.num_attention
    # ZS: args.depth
    for tmp_i in range(12):
        attn_weights.append(model.blocks[tmp_i].attn.attention_sum)
    attn_weights = torch.stack(attn_weights,dim=0)  # (num_layers, num_heads, 197, 197)
    # attn_weights: summed over batch within a machine
    # attn_num: num batches

    # Check if attn_num == num_training_imgs
    print(utils.get_world_size(), utils.get_rank(), attn_num)

    # Write.
    if utils.is_main_process():
        attn_weights = attn_weights / attn_num  # average
        print('num_vids:', attn_num)

        attn_weights_np = attn_weights.detach().cpu().numpy()
        save_path = os.path.join(args.output_dir, 'attention_score_pruned.npy')
        with open(save_path, "wb") as f:
            np.save(f, attn_weights_np)