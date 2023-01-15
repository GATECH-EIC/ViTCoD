import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys
import numpy as np


axis_sparsity = [10,30,50,70,90,100]

k_means_sparsity = [13, 17, 21.5, 27.5, 35.5, 40.5, 46, 52.5, 59, 65]
k_means_BLEU = [34.5, 34.7, 34.8, 35, 34.75, 34.3, 34.1, 33.9, 33, 29.3]

quant_sparsity = [12, 15, 19, 24, 30, 34, 38, 42.5, 47, 52, 57.5]
quant_BLEU = [34.2, 34.3, 34.4, 34.45, 34.8, 34.3, 34, 33.5, 32.6, 29.5, 31.25]

routing_sparsity = [14, 18, 22.5, 29, 37, 41.5, 47, 53, 59.5, 67]
routing_BLEU = [34.2, 34.6, 34.55, 34.73, 34, 33.8, 33, 32.5, 30.05, 23.25]

longformer_sparsity = [16.5, 21, 27.5, 35, 39.5, 45, 52, 59, 66.5]
longformer_BLEU = [34.5, 34.68, 34.75, 34.5, 34.52, 34, 33.55, 31.75, 27.25]

vit_sparsity = [35, 61, 78, 90, 92, 94, 96, 98]
vit_acc = [81.65, 81.46, 81.2, 80.78, 80.48, 80, 79.6, 79.2,]

colors = ['purple','g','b','r','c', 'y', 'k']
marker = ['^', 's', 'p', 'o', '*', '>', 'h'] 
font_big = 54
font_mid = 48
font_leg = 36
font_small = 40
lw = 8

# ax is for NLP transformer
fig, ax = plt.subplots(1,1,figsize=(26, 13))
plt.subplots_adjust(wspace=0.1, hspace=0.35)
# ax2 is for vit
ax2 = ax.twinx()

ax.plot(k_means_sparsity[:], k_means_BLEU, c='royalblue', marker=marker[3], markersize=4*lw, label=r"$\bf{NLP-Sf. k-means}$", lw=lw)
ax.plot(quant_sparsity[:], quant_BLEU, c='orange', marker=marker[3], markersize=4*lw, label=r"$\bf{NLP-Sf. quant}$", lw=lw)
ax.plot(routing_sparsity[:], routing_BLEU, c='g', marker=marker[3], markersize=4*lw, label=r"$\bf{NLP-Routing}$", lw=lw)
ax.plot(longformer_sparsity[:], longformer_BLEU, c='purple', marker=marker[3], markersize=4*lw, label=r"$\bf{NLP-Longformer}$", lw=lw)

ax.plot(vit_sparsity[:], vit_acc, c=colors[3], marker=marker[0], markersize=4*lw, label=r"$\bf{Deit-Info-Pruning}$", lw=lw)
ax2.plot(vit_sparsity[:], vit_acc, c=colors[3], marker=marker[0], markersize=4*lw, label=r"$\bf{Deit-Info-Pruning)}$", lw=lw)

ax.fill_between(axis_sparsity, 0, 1, where=((np.array(axis_sparsity)<90) & (np.array(axis_sparsity)>30)),
    color='green', alpha=0.2, transform=ax.get_xaxis_transform())

ax2.fill_between(axis_sparsity, 0, 1, where=(np.array(axis_sparsity)>70),
    color='red', alpha=0.2, transform=ax2.get_xaxis_transform())

ax.set_ylim([22, 38])
ax2.set_ylim([78.5, 82])

ax.set_xlabel('Sparsity Ratio (%)', fontsize=font_mid, fontweight="bold", y=0.95)
ax.set_ylabel('BLEU', fontsize=font_mid, fontweight="bold", rotation=90, labelpad=10)
ax2.set_ylabel('Accuracy (%)', fontsize=font_mid, fontweight="bold", rotation=270, labelpad=60)

ax.legend(fontsize=font_mid)

# custom func for set xscale
def forward(x):
    return x**(1.3)

def inverse(x):
    return x**(1/1.3)

ax.set_xscale('function', functions=(forward, inverse))
ax.set_xticks(axis_sparsity)
ax.xaxis.set_tick_params(labelsize=font_small)
ax.set_xticklabels(axis_sparsity)
ax.yaxis.set_tick_params(labelsize=font_small)
ax2.yaxis.set_tick_params(labelsize=font_small)

ax.spines['bottom'].set_linewidth(6)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(6)
ax.spines['left'].set_color('black')
ax.spines['top'].set_linewidth(6)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(6)
ax.spines['right'].set_color('black')

leg = ax.legend(fontsize=font_leg, ncol=2, loc='lower left')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(5)
plt.grid()
plt.savefig('sparsity_nlp_vs_vit.png')
# plt.savefig('sparsity_nlp_vs_vit.pdf')
# plt.savefig('sparsity_nlp_vs_vit.svg')