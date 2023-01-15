# ViTCoD Simulator (Sparse)

import numpy as np
from torch import embedding
from SRAM import SRAM
from PE import PE_array
from scipy.sparse import coo_matrix
import logging
import os
import math

# root = 'masks/deit/deit_small_lowrank'
# sparse = [0.5]
# embedding = 192
# root = 'masks/levit/LeViT_192_lowrank/0.5'
# root = '/home/sheminghao/shh/ViTCoD/attention_mask'
root = 'masks/deit_tiny_lowrank'
sparse = [0.95]
total_preload_linear_cycles = 0
total_preload_ffn_cycles = 0
total_linear_PE_cycles = 0
total_ffn_PE_cycles = 0
total_PRE_cycles = 0

log = logging.getLogger()
# TODO:
log_path = os.path.join(root, 'vitcod_atten_ffn.txt')
handlers = [logging.FileHandler(log_path, mode='a+'),
            logging.StreamHandler()]
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=handlers)
len_gb = 0
len_sparse = 0

for p in sparse:
    # Initialize Q, K, V and attn maps
    # TODO: load the masks of attention and global tokens
    # attn_map_mask = np.load(root+'/reodered_info_'+str(p)+'.npy')
    # num_global_tokens = np.load(root+'/global_token_info_'+str(p)+'.npy')
    attn_map_mask = np.load(root+'/reodered_info_'+str(p)+'.npy')
    num_global_tokens = np.load(root+'/global_token_info_'+str(p)+'.npy')
    # attn_map_mask = np.load(root+'/reodered_mask_'+str(p)+'.npy')
    # num_global_tokens = np.load(root+'/global_token_mask_'+str(p)+'.npy')
    all_Q = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], 64))
    all_K = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], 64))
    all_V = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], 64))
    log.info('Shape: {}'.format(all_V.shape))
    my_SRAM = SRAM()
    my_PE = PE_array()

    head = all_Q.shape[1]
    # TODO: the compression ratio
    ratio = 2/3
    embedding = 192
    # if attn_map_mask.shape[1] == 3:
    #     ratio = 2/3
    # elif attn_map_mask.shape[1] == 5:
    #     ratio = 3/5
    # elif attn_map_mask.shape[1] == 6:
    #     ratio = 4/6
    # if p == 'dpt0':
    #     embedding = 192
    # elif p == 'dpt1':
    #     embedding = 288
    # elif p == 'dpt2':
    #     embedding = 384
    
    PE_width = 64
    PE_height = 8
    

    for _layer in range(all_Q.shape[0]):
        
        for _head in range(head//head):

            Q = all_Q[_layer, _head]
            K = all_K[_layer, _head]
            V = all_V[_layer, _head]
            log.info('***' * 10)
            log.info('Layer: {}; Head: {}'.format(_layer, _head))
            # print('***' * 10)

            # ############## embedding ##############
            # K-stationary (Why? Because the number of gloal tokens vary a lot --> Score stationary is not best fit)
            preload_cycles = 0
            PRE_cycles = 0
            SDDMM_PE_cycles = 0
            # ############ Q #########
            for _sta_q in range(Q.shape[0]): 
                if _sta_q == 0:
                    for f in range(Q.shape[1]):
                        preload_cycles += my_SRAM.preload_weight(nums=head*1*embedding, bits=8, bandwidth_ratio=1)
            
                for k in range(int((embedding* Q.shape[1])//int(PE_width*PE_height/head))):
                        SDDMM_PE_cycles += 1

            # ############ K #########
            for _sta_q in range(K.shape[0]): 
                if _sta_q == 0:
                    for f in range(K.shape[1]):
                        preload_cycles += my_SRAM.preload_weight(nums=head*1*embedding, bits=8, bandwidth_ratio=1)
            
                for k in range(int((embedding* K.shape[1])//int(PE_width*PE_height/head))):
                        SDDMM_PE_cycles += 1
                
            # ############ V #########
            for _sta_q in range(V.shape[0]): 
                if _sta_q == 0:
                    for f in range(V.shape[1]):
                        preload_cycles += my_SRAM.preload_weight(nums=head*1*embedding, bits=8, bandwidth_ratio=1)
            
                for v in range(int((embedding* V.shape[1])//int(PE_width*PE_height/head))):
                        SDDMM_PE_cycles += 1

            # store back to DRAM
            # process Q
            for num in range(Q.shape[0]): 
                preload_cycles += my_SRAM.preload_encoder(nums=head*1*Q.shape[1], bits=8, bandwidth_ratio=1/(head*ratio))
                # ######### Preprocessing 
                for k in range(math.ceil((head*1* Q.shape[1])//int(PE_width*PE_height/(head*ratio)))):
                    PRE_cycles += 1
                # ######### Store back 
                preload_cycles += my_SRAM.store_out(nums=1* Q.shape[1], bits=8, bandwidth_ratio=1/(head*ratio))

            # process K
            for num in range(K.shape[0]): 
                preload_cycles += my_SRAM.preload_encoder(nums=head*1*K.shape[1], bits=8, bandwidth_ratio=1/(head*ratio))
                # ######### Preprocessing 
                for k in range(math.ceil((head*1* K.shape[1])//int(PE_width*PE_height/(head*ratio)))):
                    PRE_cycles += 1
                # ######### Store back 
                preload_cycles += my_SRAM.store_out(nums=1* K.shape[1], bits=8, bandwidth_ratio=1/(head*ratio))

            # process V
            for num in range(V.shape[0]): 
                # ######### Store back 
                preload_cycles += my_SRAM.store_out(nums=1* V.shape[1], bits=8, bandwidth_ratio=1/(head))
            

            # ############# concat of multi-head #######################
            for _tile_attn in range(int(Q.shape[0]*embedding*Q.shape[1]// int(PE_height*PE_width/head))):
                SDDMM_PE_cycles += 1
            for num in range(Q.shape[0]// int(PE_height*PE_width/head)):
                for _tile_attn in range(embedding):
                    preload_cycles += my_SRAM.preload_weight(nums=Q.shape[1], bits=8, bandwidth_ratio=1/head)  

            total_preload_linear_cycles += preload_cycles
            total_PRE_cycles += PRE_cycles
            total_linear_PE_cycles += SDDMM_PE_cycles

            log.info('Embedding dataloader | cycles: {}'.format(preload_cycles))
            log.info('Embedding decoder | cycles: {}'.format(PRE_cycles))
            log.info('Embedding calcuation | cycles: {}'.format(SDDMM_PE_cycles))    
            

            log.info('***' * 4)
            SDDMM_PE_cycles = 0
            preload_cycles = 0
            # ############# FFN #######################
            for _tile_attn in range(int(Q.shape[0]*embedding*embedding*4// int(PE_height*PE_width))):
                SDDMM_PE_cycles += 1
            for num in range(Q.shape[0]// int(PE_height*PE_width)):
                for _tile_attn in range(embedding*4):
                    preload_cycles += my_SRAM.preload_weight(nums=embedding, bits=8, bandwidth_ratio=1)
            # if not 'strided' in p:
            for _tile_attn in range(int(Q.shape[0]*embedding*embedding*4// int(PE_height*PE_width))):
                SDDMM_PE_cycles += 1
            for num in range(Q.shape[0]// int(PE_height*PE_width)):
                for _tile_attn in range(embedding):
                    preload_cycles += my_SRAM.preload_weight(nums=embedding*4, bits=8, bandwidth_ratio=1)

            total_preload_ffn_cycles += preload_cycles
            total_ffn_PE_cycles += SDDMM_PE_cycles

            log.info('Embedding dataloader | cycles: {}'.format(preload_cycles))
            log.info('Embedding calcuation | cycles: {}'.format(SDDMM_PE_cycles))
           
    
log.info('')
log.info('***' * 10)
log.info('total linear preprocessing cycles: {}'.format(total_PRE_cycles))
log.info('total linear preloading cycles: {}'.format(total_preload_linear_cycles))
log.info('total linear computation cycles: {}'.format(total_linear_PE_cycles))
log.info('total ffn preloading cycles: {}'.format(total_preload_ffn_cycles))
log.info('total ffn computation cycles: {}'.format(total_ffn_PE_cycles))

log.info('')
linear = max(total_linear_PE_cycles+total_PRE_cycles,total_preload_linear_cycles)
ffn = max(total_ffn_PE_cycles ,total_preload_ffn_cycles)
log.info('total linear cycles: {}'.format(linear))
log.info('total ffn cycles: {}'.format(ffn))
log.info('total cycles: {}'.format(ffn+linear))
log.info('***' * 10)