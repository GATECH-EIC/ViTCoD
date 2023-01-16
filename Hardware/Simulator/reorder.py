import numpy as np
import matplotlib.pyplot as plt
import dgl
import torch
from dgl.distributed import partition_graph
import os  

def calc(graph, ax, threshold=90):
#     print('start new')
    a0 = graph
    # print(graph.shape)
    n = graph.shape[0]
    u_list = []
    v_list = []
    for i in range(n):
        for j in range(n):
            if not a0[i][j]:
                u_list.append(j)
                v_list.append(i)
    g = dgl.graph((u_list, v_list))
    g.ndata['in_deg'] = g.in_degrees()

    n_node = g.num_nodes()
    n_edge = g.num_edges()
    # avg_edge = n_edge / tot_subgraphs
#     print(g.edges(), n_edge)
    # edge_list =
    out_deg = g.out_degrees()
    high_density = out_deg[out_deg > threshold]
    high_density_idx = np.where(out_deg > threshold)[0]

    total = len(high_density_idx)

    tmp1 = 200
    tmp2 = 300
    orig_a, orig_b = g.edges()
#     print(total)
    total_dense = 0
    for i in high_density_idx:
        total_dense += torch.sum(orig_a == i)

    for i in range(total):
        orig_a[orig_a == i] = tmp1
        orig_b[orig_b == i] = tmp1
        orig_a[orig_a == high_density_idx[i]] = i
        orig_b[orig_b == high_density_idx[i]] = i
#         print(torch.tensor(high_density_idx[i]))
        orig_a[orig_a == tmp1] = torch.tensor(high_density_idx[i])
        orig_b[orig_b == tmp1] = torch.tensor(high_density_idx[i])
    dense_cnt = total_dense

    new_graph = torch.ones(graph.shape[0],graph.shape[1])
    for i in range(len(orig_a)):
        new_graph[orig_b[i], orig_a[i]] = 0
    new_graph = new_graph.numpy()
    # ax.imshow(new_graph, cmap='viridis') # , interpolation='none')
    # ax.axis('off')
    total_cnt = n_edge
    return dense_cnt, total_cnt, new_graph, total

       
# def getfile_name(file_dir):   
#     for root, dirs, files in os.walk(file_dir):  
#         print(root) #当前目录路径  
#         print(dirs) #当前路径下所有子目录  
#         print(files) #当前路径下所有非目录子文件  

# getfile_name('masks')
for root, dirs, files in os.walk('/home/sheminghao/ViTCoD/Hardware/Simulator/masks/deit_tiny_lowrank'):
    print('files', files)
    files = ['info_0.95.npy']
    for file in files: 
        mask = np.load(root+'/'+file)
        print(mask.shape)

        # before reorder
        # fig, ax = plt.subplots(12,12, figsize=[20,20])
        # for i in range(12):
        #     for j in range(12):
        #         ax[i, j].imshow(mask[i, j], cmap='viridis')
        #         ax[i, j].axis('off')
        # plt.savefig('data/attn.png', bbox_inches='tight')
        # plt.close()

        # after reorder
        fig, ax = plt.subplots(mask.shape[0]+1,mask.shape[1], figsize=[20,20])
        D = 0
        E = 0
        new_mask = np.zeros(mask.shape)
        num_global_tokens = np.zeros((mask.shape[0], mask.shape[1]))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                cnt_d, cnt_e, _new_mask, _num_global_tokens = calc(mask[i,j], ax[i,j], threshold=50)
                new_mask[i, j] = _new_mask
                num_global_tokens[i, j] = _num_global_tokens
                D += cnt_d
                E += cnt_e
        print('Dense: {} ({:.2f}%), Sparse: {} ({:.2f}%), Total: {}'.format(\
            D, D/E*100, E-D, (E-D)/E*100, E))
        print('Overall Sparisty: {:.2f}%'.format(100 - E/12/12/197/197*100))

        np.save(root+'/reodered_'+file, new_mask)
        np.save(root+'/global_token_'+file, num_global_tokens)

    # plt.savefig('data/attn_reorder_50.png', bbox_inches='tight')