import numpy as np
import os
import scipy.sparse

import argparse

def gen_random_mask(attn_map_path, sparsity_ratio):
    attn_map = np.load(attn_map_path)
    num_layers, num_heads, num_tokens, _ = attn_map.shape
    sparse_attn_map = []
    density = 1 - sparsity_ratio

    for i in range(num_layers):
        sparse_attn_map.append([np.ceil(scipy.sparse.random(num_tokens, num_tokens, density=density).A) for _ in range(num_heads)])
    sparse_attn_map = np.asarray(sparse_attn_map).astype(int).astype(bool)

    save_path = os.path.join('', 'no_mask.npy')
    with open(save_path, "wb") as f:
        np.save(f, sparse_attn_map)


def gen_ratio_based_mask(attn_map_path, sparsity_ratio, output_dir):
    attn_map = np.load(attn_map_path)
    num_layers, num_heads, num_tokens, _ = attn_map.shape
    sparse_attn_map = []
    cut_off = sparsity_ratio * num_tokens

    for i in range(num_layers):
        for j in range(num_heads):
            data = attn_map[i][j]
            order = np.argsort(data)
            rank = np.argsort(order)
            mask = rank < cut_off
            mask[0, :] = 0
            mask[:, 0] = 0
            sparse_attn_map.append(mask)
    sparse_attn_map = np.asarray(sparse_attn_map)
    sparse_attn_map = sparse_attn_map.reshape(num_layers, num_heads, num_tokens, num_tokens)
    print ("Sparsity: ", sparsity_ratio)
    print ("TotalAve Sparsity: ", np.count_nonzero(sparse_attn_map) / (12*12*197*197))

    sparsity = np.round(np.count_nonzero(sparse_attn_map) / (12*12*197*197), 2)

    # save_path = os.path.join('', 'ratio_based_mask_95.npy')
    save_path = os.path.join(output_dir, 'ratio_{}.npy'.format(sparsity))
    with open(save_path, "wb") as f:
        np.save(f, sparse_attn_map)



def gen_info_based_mask(attn_map_path, info, output_dir):
    attn_map = np.load(attn_map_path)
    # print (attn_map[0][0][0])
    # attn_map = np.random.randint(10, size=(1, 1, 5, 5))
    # print (attn_map)
    num_layers, num_heads, num_tokens, _ = attn_map.shape
    sparse_attn_map = []

    for i in range(num_layers):
        for j in range(num_heads):
            temp = []
            for k in range(num_tokens):
                arr = attn_map[i][j][k]
                temp.append(info_cutoff(arr, info))
            temp = np.asarray(temp)
            temp = temp.reshape(num_tokens, num_tokens)
            temp[0, :] = 0
            temp[:, 0] = 0
            sparse_attn_map.append(temp)
    sparse_attn_map = np.asarray(sparse_attn_map)
    sparse_attn_map = sparse_attn_map.reshape(num_layers, num_heads, num_tokens, num_tokens)
    # print (sparse_attn_map)
    # print (sparse_attn_map[0][0][0])
    sparsity = 1.0
    print ("Info Cut-off: ", info)
    if 'base' in output_dir:
        print ("TotalAve Sparsity: ", np.count_nonzero(sparse_attn_map) / (12*12*197*197))
        sparsity = np.round(np.count_nonzero(sparse_attn_map) / (12*12*197*197), 2)
    elif 'small' in output_dir:
        print ("TotalAve Sparsity: ", np.count_nonzero(sparse_attn_map) / (6*12*197*197))
        sparsity = np.round(np.count_nonzero(sparse_attn_map) / (6*12*197*197), 2)
    elif 'tiny' in output_dir:
        print ("TotalAve Sparsity: ", np.count_nonzero(sparse_attn_map) / (3*12*197*197))
        sparsity = np.round(np.count_nonzero(sparse_attn_map) / (3*12*197*197), 2)

    save_path = os.path.join(output_dir, 'info_{}.npy'.format(sparsity))
    with open(save_path, "wb") as f:
        np.save(f, sparse_attn_map)


def info_cutoff(arr, info):
    sorted_arr = np.sort(arr)[::-1]
    # print (sorted_arr)
    sum = 0.0
    order = np.argsort(arr)
    rank = np.argsort(order)
    idx = 0
    while sum < info and idx < arr.shape[0]:
        sum += sorted_arr[idx]
        idx += 1
    idx = arr.shape[0] - idx - 2
    # print (idx)
    mask = rank <= idx
    # print (mask)
    return mask


def gen_std_based_mask(attn_map_path, std_coef):
    attn_map = np.load(attn_map_path)
    num_layers, num_heads, num_tokens, _ = attn_map.shape
    sparse_attn_map = []

    for i in range(num_layers):
        for j in range(num_heads):
            data = attn_map[i][j]
            mean = np.mean(data, axis=1)
            # print (mean.shape)
            std = np.std(data, axis=1)
            # print ((mean + std_coef * std).transpose().shape)
            cut_off = np.transpose(np.tile((mean + std_coef * std), (num_tokens, 1)))
            # print (cut_off)
            mask = data < cut_off
            sparse_attn_map.append(mask)
    sparse_attn_map = np.asarray(sparse_attn_map)
    sparse_attn_map = sparse_attn_map.reshape(num_layers, num_heads, num_tokens, num_tokens)
    print (np.count_nonzero(sparse_attn_map) / (12*12*197*197))
    save_path = os.path.join('', 'std_based_mask_9.npy')
    with open(save_path, "wb") as f:
        np.save(f, sparse_attn_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('gen_mask')
    parser.add_argument('--attn', default='./attn/deit_base/attention_score.npy', type=str)
    parser.add_argument('--sparsity', default=0.9, type=float)
    parser.add_argument('--method', default='info', choices=['info', 'ratio', 'random', 'std'], type=str)
    parser.add_argument('--info_cut', default=0.186, type=float)
    parser.add_argument('--output_dir', default='', help='path where to save')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.method == 'info':
        gen_info_based_mask(args.attn, args.info_cut, output_dir=args.output_dir)
    elif args.method == 'ratio':
        gen_ratio_based_mask(args.attn, args.sparsity, output_dir=args.output_dir)
    elif args.method == 'random':
        gen_random_mask(args.attn, args.sparsity)
    else:
        gen_std_based_mask(args.attn, args.sparsity)