import numpy as np
import sys
import os
# np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

mask0 = np.load("attn/LeViT_128/attention_score_dpt0.npy")
subsample0 = np.load("attn/LeViT_128/attention_score_subsample0.npy")
mask1 = np.load("attn/LeViT_128/attention_score_dpt1.npy")
subsample1 = np.load("attn/LeViT_128/attention_score_subsample1.npy")
mask2 = np.load("attn/LeViT_128/attention_score_dpt2.npy")

test = np.load("masks/LeViT_128/test_subsample1.npy")
print(test.shape)
print(mask0.shape)
print(subsample0.shape)
print(mask1.shape)
print(subsample1.shape)
print(mask2.shape)

test_mask0 = np.zeros_like(mask0)
test_subsample0 = np.zeros_like(subsample0)
test_mask1 = np.zeros_like(mask1)
test_subsample1 = np.zeros_like(subsample1)
test_mask2 = np.zeros_like(mask2)

# with open("masks/LeViT_128/test_mask0.npy", "wb") as f:
#     np.save(f, test_mask0)
# with open("masks/LeViT_128/test_subsample0.npy", "wb") as f:
#     np.save(f, test_subsample0)
# with open("masks/LeViT_128/test_mask1.npy", "wb") as f:
#     np.save(f, test_mask1)
# with open("masks/LeViT_128/test_subsample1.npy", "wb") as f:
#     np.save(f, test_subsample1)
# with open("masks/LeViT_128/test_mask2.npy", "wb") as f:
#     np.save(f, test_mask2)