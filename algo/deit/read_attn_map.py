import numpy as np
import sys
import os
# np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

# arr = np.load('/home/zs19/deit/info_based_mask_9.npy')
# tmp = arr[11][0]
# vals = []
# for i in range(12):
#     for j in range(12):
#         vals.append(arr[i][j])

# fig = plt.figure()

# grid = AxesGrid(fig, 111,
#                 nrows_ncols=(12, 12),
#                 axes_pad=0.015,
#                 share_all=True,
#                 label_mode="L",
#                 cbar_location="right",
#                 cbar_mode="single",
#                 )

# for val, ax in zip(vals,grid):
#     im = ax.imshow(1-val, cmap='Greys')

# grid.cbar_axes[0].colorbar(im)

# for cax in grid.cbar_axes:
#     cax.toggle_label(False)

# plt.savefig("info_mask_vis")

arr = np.load('/home/zs19/deit/info_based_mask_9.npy')
print (np.count_nonzero(arr) / (144 * 197 * 197))
# tmp = arr[9][1]
# print (tmp[0, :])
# plt.imshow(1-tmp, cmap='Greys',  interpolation='nearest')
# plt.figtext(0.5, 0.01, "Head 9-1 Ave. Attn.", wrap=True, horizontalalignment='center', fontsize=12)
# plt.savefig("info_mask_9_1")