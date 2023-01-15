import numpy as np
import torch
import torch.nn as nn

from utils import is_main_process


def cal_reduced_Gflops(pattern_mask, L, C):
    '''
    Return reduced_number_of_GFlops thanks to patterns
    '''
    num_total_all, num_mask_all, _, _ = \
                    cal_num_elements_pattern(pattern_mask)
    mask_ratio_all_norm = num_mask_all * 1.0 / num_total_all
    print('Masking element raio: %.2f%%' % (mask_ratio_all_norm*100))

    N = pattern_mask[0].size(2)
    reduced_GFLOPs = (2 * mask_ratio_all_norm * N * N * C) / 1e9
    print('Reduced number of GFlops per Layer: %.4f' % (reduced_GFLOPs))
    reduced_GFLOPs *= L
    print('Reduced number of GFlops in total: %.4f' % (reduced_GFLOPs))

    return reduced_GFLOPs


def return_mask_size(train_size_thw, patch_size, num_heads):
    (train_t, train_h, train_w) = train_size_thw
    p_train_h, p_train_w = train_h // patch_size, train_w // patch_size
    n_tokens = train_t * p_train_h * p_train_w
    n_tokens += 1  # cls_token

    mask_size_np = (1, num_heads, n_tokens, n_tokens)
    mask_size_pt = torch.Size(mask_size_np)
    mask_size_wo_cls = (1, num_heads, n_tokens-1, n_tokens-1)
    mask_size_wo_cls_pt = torch.Size(mask_size_wo_cls)

    return mask_size_np, mask_size_pt, mask_size_wo_cls, mask_size_wo_cls_pt


def cal_num_elements_pattern(pattern_mask):
    '''
    Return num_(0,1)_elements in the given pattern.
    0 (False): remaining elements
    1 (True): elements to be masked out.
    '''
    num_total_all, num_mask_all = 0, 0
    num_total_wo_cls, num_mask_wo_cls = 0, 0
    for each_pattern in pattern_mask:
        # each_pattern: (1, num_heads, thw+1, thw+1)
        num_total_all += torch.numel(each_pattern)
        num_mask_all += each_pattern.sum().item()

        num_total_wo_cls += torch.numel(each_pattern[:,:,1:,1:])
        num_mask_wo_cls += each_pattern[:,:,1:,1:].sum().item()

    return num_total_all, num_mask_all, num_total_wo_cls, num_mask_wo_cls


def print_pattern_ratio(pattern_mask):
    '''
    Print total ratio of remaining elements after masking.
    0 (False): remaining elements
    1 (True): elements to be masked out.
    '''
    num_total_all, num_mask_all, num_total_wo_cls, num_mask_wo_cls = \
                    cal_num_elements_pattern(pattern_mask)

    num_remain_all = num_total_all - num_mask_all
    print('Num total elements: %d, Num remaining elements: %d' % \
            (num_total_all, num_remain_all))
    ramain_ratio_all = num_remain_all*1.0 / num_total_all * 100
    print('#'*30 + \
            'Ratio of remaining elements: %.2f%%' % \
            (ramain_ratio_all) + '#'*30)
    num_remain_wo_cls = num_total_wo_cls - num_mask_wo_cls
    ramain_ratio_wo_cls = num_remain_wo_cls*1.0 / num_total_wo_cls * 100
    print('Num total elements: %d, Num remaining elements excluding cls: %d' % \
            (num_total_wo_cls, num_remain_wo_cls))
    print('#'*30 + \
            'Ratio of remaining elements excluding cls token: %.2f%%' % \
            (ramain_ratio_wo_cls) + '#'*30)


def mask_read_files(train_size_thw, patch_size,
                    num_heads, num_layers, filepath,
                    ):
    _, size_pt, _, _ = \
    return_mask_size(train_size_thw, patch_size, num_heads)

    # Load npy, size of (num_layers, num_heads, thw+1, thw+1)
    if is_main_process():
        print('#'*50 + 'Load a mask from an external file.' + '#'*50)
        print('#'*50 + filepath + '#'*50)
    with open(filepath, "rb") as f:
        mask_loaded = np.load(f)

    pattern_mask = []
    # Different patterns with different layers.
    for seed in range(num_layers):
        # (num_heads, thw+1, thw+1)
        pattern_mask_shared = torch.from_numpy(mask_loaded[seed])
        pattern_mask_shared = pattern_mask_shared.unsqueeze(0)
        assert pattern_mask_shared.size() == size_pt
        pattern_mask.append(pattern_mask_shared)
    return pattern_mask