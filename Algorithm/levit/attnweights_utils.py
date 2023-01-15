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
    for tmp_i in range(14):
        if tmp_i == 4 or tmp_i == 9:
            model.blocks[tmp_i * 2].num_attention.fill_(0)
            model.blocks[tmp_i * 2].attention_sum.fill_(0)
        else:
            model.blocks[tmp_i * 2].m.num_attention.fill_(0)
            model.blocks[tmp_i * 2].m.attention_sum.fill_(0)

    # for tmp_i in range(5, 9):
    #     model.blocks[tmp_i * 2].m.num_attention.fill_(0)
    #     model.blocks[tmp_i * 2].m.attention_sum.fill_(0)

    # for tmp_i in range(10, 14):
    #     model.blocks[tmp_i * 2].m.num_attention.fill_(0)
    #     model.blocks[tmp_i * 2].m.attention_sum.fill_(0)


def save(model, args):
    attn_weights_0 = []
    attn_weights_subsample_0 = []
    attn_weights_1 = []
    attn_weights_subsample_1 = []
    attn_weights_2 = []
    attn_num = model.blocks[0].m.num_attention
    for tmp_i in range(4):
        attn_weights_0.append(model.blocks[tmp_i * 2].m.attention_sum)
    attn_weights_subsample_0.append(model.blocks[4 * 2].attention_sum)
    for tmp_i in range(5, 9):
        attn_weights_1.append(model.blocks[tmp_i * 2].m.attention_sum)
    attn_weights_subsample_1.append(model.blocks[9 * 2].attention_sum)
    for tmp_i in range(10, 14):
        attn_weights_2.append(model.blocks[tmp_i * 2].m.attention_sum)
    
    
    attn_weights_0 = torch.stack(attn_weights_0,dim=0)  # (num_layers, num_heads, 197, 197)
    attn_weights_subsample_0 = torch.stack(attn_weights_subsample_0,dim=0) 
    attn_weights_1 = torch.stack(attn_weights_1,dim=0) 
    attn_weights_subsample_1 = torch.stack(attn_weights_subsample_1,dim=0) 
    attn_weights_2 = torch.stack(attn_weights_2,dim=0) 
    # attn_weights: summed over batch within a machine
    # attn_num: num batches

    # Check if attn_num == num_training_imgs
    print(utils.get_world_size(), utils.get_rank(), attn_num)

    # Write.
    if utils.is_main_process():
        attn_weights_0 = attn_weights_0 / attn_num  # average
        attn_weights_subsample_0 = attn_weights_subsample_0 / attn_num
        attn_weights_1 = attn_weights_1 / attn_num
        attn_weights_subsample_1 = attn_weights_subsample_1 / attn_num
        attn_weights_2 = attn_weights_2 / attn_num
        print('num_vids:', attn_num)

        attn_weights_np_0 = attn_weights_0.detach().cpu().numpy()
        attn_weights_subsample_0 = attn_weights_subsample_0.detach().cpu().numpy()
        attn_weights_np_1 = attn_weights_1.detach().cpu().numpy()
        attn_weights_subsample_1 = attn_weights_subsample_1.detach().cpu().numpy()
        attn_weights_np_2 = attn_weights_2.detach().cpu().numpy()
        
        save_path_0 = os.path.join(args.output_dir, 'attention_score_dpt0.npy')
        save_path_subsample_0 = os.path.join(args.output_dir, 'attention_score_subsample0.npy')
        save_path_1 = os.path.join(args.output_dir, 'attention_score_dpt1.npy')
        save_path_subsample_1 = os.path.join(args.output_dir, 'attention_score_subsample1.npy')
        save_path_2 = os.path.join(args.output_dir, 'attention_score_dpt2.npy')
        
        # print (save_path)
        with open(save_path_0, "wb") as f:
            np.save(f, attn_weights_np_0)
        with open(save_path_subsample_0, "wb") as f:
            np.save(f, attn_weights_subsample_0)
        with open(save_path_1, "wb") as f:
            np.save(f, attn_weights_np_1)
        with open(save_path_subsample_1, "wb") as f:
            np.save(f, attn_weights_subsample_1)
        with open(save_path_2, "wb") as f:
            np.save(f, attn_weights_np_2)
