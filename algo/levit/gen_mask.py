import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
import argparse


def gen_attn_paths(attn_dir):
    load_path = {'dpt0': os.path.join(attn_dir, 'attention_score_dpt0.npy'),
                 'subsample0': os.path.join(attn_dir, 'attention_score_subsample0.npy'),
                 'dpt1': os.path.join(attn_dir, 'attention_score_dpt1.npy'),
                 'subsample1': os.path.join(attn_dir, 'attention_score_subsample1.npy'),
                 'dpt2': os.path.join(attn_dir, 'attention_score_dpt2.npy'),}
    return load_path


def gen_mask_paths(output_dir):
    save_mask_path = {'dpt0': os.path.join(output_dir, 'mask_dpt0.npy'),
                 'subsample0': os.path.join(output_dir, 'mask_subsample0.npy'),
                 'dpt1': os.path.join(output_dir, 'mask_dpt1.npy'),
                 'subsample1': os.path.join(output_dir, 'mask_subsample1.npy'),
                 'dpt2': os.path.join(output_dir, 'mask_dpt2.npy'),}
    return save_mask_path


def gen_info_based_mask(attn_dir, output_dir, info_cut_map):
    load_path = gen_attn_paths(attn_dir)

    stage_to_attention_map = {}
    stage_to_shape_map = {}
    stage_to_mask_map = {}
    stage_to_sparsity_map = {}

    for (k,v) in load_path.items():
        stage_to_attention_map[k] = np.load(v)
        stage_to_shape_map[k] = stage_to_attention_map[k].shape
        print(k, stage_to_attention_map[k].shape)

    for (stage,attn_map) in stage_to_attention_map.items(): 
        stage_to_mask_map[stage] = np.apply_along_axis(info_cutoff, 3, arr=attn_map, info_cut_map=info_cut_map, stage=stage)
    
    total_nonzero = 0
    total_num = 0

    for (stage,mask_map) in stage_to_mask_map.items(): 
        (n_layer, n_head, win_size_h, win_size_w) = stage_to_shape_map[stage]
        nonzero = np.count_nonzero(mask_map)
        num = n_layer * n_head * win_size_h * win_size_w
        # print(num)
        total_nonzero += nonzero
        total_num += num
        stage_to_sparsity_map[stage] = nonzero / num
        print (stage, stage_to_sparsity_map[stage])
        print(nonzero, num)
    
    print ("Total sparsity: ", total_nonzero / total_num)
    total_sparsity = np.round(total_nonzero / total_num, 2)
    output_dir += str(total_sparsity)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    save_path = gen_mask_paths(output_dir)
    for (stage, save_path) in save_path.items():
        with open(save_path, "wb") as f:
            np.save(f, stage_to_mask_map[stage])


def info_cutoff(arr, info_cut_map, stage):
    info = info_cut_map[stage]
    sorted_arr = np.sort(arr)[::-1]
    sum = 0.0
    order = np.argsort(arr)
    rank = np.argsort(order)
    idx = 0
    while sum < info and idx < arr.shape[0]:
        sum += sorted_arr[idx]
        idx += 1
    idx = arr.shape[0] - idx - 2
    mask = rank <= idx
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser('gen_mask')
    parser.add_argument('--attn_path', default='attn/LeViT_128_lowrank', type=str)
    parser.add_argument('--info_cut_dpt0', default=0.313, type=float)
    parser.add_argument('--info_cut_subsample0', default=0.53, type=float)
    parser.add_argument('--info_cut_dpt1', default=0.16, type=float)
    parser.add_argument('--info_cut_subsample1', default=0.33, type=float)
    parser.add_argument('--info_cut_dpt2', default=0.6, type=float)
    parser.add_argument('--output_dir', default='./masks/LeViT_128_lowrank/', help='path where to save')
    args = parser.parse_args()

    info_cut_map = {'dpt0': args.info_cut_dpt0,
                 'subsample0': args.info_cut_subsample0,
                 'dpt1': args.info_cut_dpt1,
                 'subsample1': args.info_cut_subsample1,
                 'dpt2': args.info_cut_dpt2}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # gen_info_based_mask(args.attn_path, args.output_dir, info_cut_map)
