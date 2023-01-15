import numpy as np
import sys
import os
# np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import argparse

# arr = np.load('/home/zs19/LeViT_copy/info_based_mask_90_dpt3.npy')

def plot_func(arr, save_dir):
    print (arr.shape)
    # print (arr[0][0])
    vals = []
    num_heads = arr.shape[1]
    num_layers = arr.shape[0]
    for i in range(num_layers):
        for j in range(num_heads):
            vals.append(arr[i][j])


    # print (vals[0][0])
    fig = plt.figure()

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(num_layers, num_heads),
                    axes_pad=0.015,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    for val, ax in zip(vals,grid):
        im = ax.imshow(val, vmin=0, vmax=1)
        # im = ax.imshow(1-val)

    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    fig.tight_layout()
    plt.savefig(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('visualize attn')
    parser.add_argument('--attn', default='attn/LeViT_128/attention_score_subsample0.npy', type=str)
    parser.add_argument('--save_dir', default='attn/LeViT_128/subsample0')

    args = parser.parse_args()

    plot_func(np.load(args.attn), args.save_dir)