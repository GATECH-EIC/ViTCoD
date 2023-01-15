# ViTCoD Simulator (Sparse)

import numpy as np
from SRAM import SRAM
from PE import PE_array
from scipy.sparse import coo_matrix
import logging
import os
import math

root = 'masks/deit_tiny_lowrank'
# root = 'masks/reorder/deit/reorder_att/deit_base'
sparse = [0.95]

for p in sparse:
    # Logging
    log = logging.getLogger()
    log_path = os.path.join(root, 'vitcod_atten_'+str(p)+'_wo.txt')
    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    # Initialize Q, K, V and attn maps
    attn_map_mask = np.load(root+'/reodered_info_'+str(p)+'.npy')
    num_global_tokens = np.load(root+'/global_token_info_'+str(p)+'.npy')
    
    # dim of (layer, head, token, features)
    all_Q = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], 64))
    all_K = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], 64))
    all_V = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], 64))
    log.info('Shape: {}'.format(all_V.shape))
    my_SRAM = SRAM()
    my_PE = PE_array()

    total_preload_cycles = 0
    total_SDDMM_PE_cycles = 0
    total_SpMM_PE_cycles = 0
    total_linear_PE_cycles = 0
    total_PRE_cycles = 0

    head = all_Q.shape[1]
    # the compression ratio of head via the encoder
    ratio = 2/3
    PE_width = 64
    PE_height = 8

    total_sparse_ratio = 0
    for _layer in range(all_Q.shape[0]):
        # head parallel
        for _head in range(head//head):

            Q = all_Q[_layer, _head]
            K = all_K[_layer, _head]
            V = all_V[_layer, _head]
            log.info('***' * 10)
            log.info('Layer: {}; Head: {}'.format(_layer, _head))

            mask = attn_map_mask[_layer, _head]
            global_tokens = int(num_global_tokens[_layer, _head])
            log.info('global tokens: {}'.format(global_tokens))
            sparser = coo_matrix(1 - mask[:, global_tokens:])
            sparser = np.column_stack((sparser.row, sparser.col))
            if global_tokens == mask.shape[1]:
                sparse_ratio = 0
            else:
                sparse_ratio = len(sparser)/(mask[:, global_tokens:].shape[0]*mask[:, global_tokens:].shape[1])
            total_sparse_ratio += sparse_ratio
            # log.info('number of non-zeros in the sparser region: {}'.format(len(sparser)))

            # data loading and pre-processing
            # ############## dense pattern q*k ##############
            preload_cycles = 0
            PRE_cycles = 0
            SDDMM_PE_cycles = 0
            for _sta_k in range(global_tokens): 
                # ############ k #########
                # ######### Load k and decoder weight
                preload_cycles += my_SRAM.preload_K(nums=head*ratio*1* K.shape[1], bits=8, bandwidth_ratio=1)
                preload_cycles += my_SRAM.preload_decoder(nums=head*ratio*1* K.shape[1], bits=8, bandwidth_ratio=1/head)
                # ######### Preprocessing 
                for k in range((math.ceil((head*ratio*1* K.shape[1])/int(PE_width*PE_height/head)))):
                    PRE_cycles += 1
                for _sta_q in range(int(Q.shape[0])):
                    if _sta_k == 0:
                    # ############ q #########
                    # ######### Load q and decoder weight
                        # reload_ratio = (Q.shape[0]-(my_SRAM.max_Q/(8*Q.shape[1]*head)))/Q.shape[0]
                        reload_ratio = 0
                        preload_cycles += my_SRAM.preload_Q(nums=head*ratio*1* Q.shape[1], bits=8, bandwidth_ratio=1)*(1+reload_ratio)
                        preload_cycles += my_SRAM.preload_decoder(nums=head*ratio*1* Q.shape[1], bits=8, bandwidth_ratio=1/head)*(1+reload_ratio)
                        # ######### Preprocessing 
                        for q in range(math.ceil((head*ratio*1* Q.shape[1])/int(PE_width*PE_height/head))):
                            PRE_cycles += 1*(1+reload_ratio)
            
            total_PRE_cycles += PRE_cycles
            total_preload_cycles += preload_cycles
            log.info('Dense SpMM dataloader | cycles: {}'.format(preload_cycles))
            log.info('Dense SpMM decoder | cycles: {}'.format(PRE_cycles))

            # ############## sparse pattern q*k ##############
            # K-stationary (Why? Because the number of gloal tokens vary a lot --> Score stationary is not best fit)
            preload_cycles = 0
            PRE_cycles = 0
            # ############ k #########
            # ######### Load K and decoder weights
            for i in range(K.shape[0]-global_tokens):
                preload_cycles += my_SRAM.preload_K(nums=head*ratio*1* K.shape[1], bits=8, bandwidth_ratio=1)
                preload_cycles += my_SRAM.preload_decoder(nums=head*ratio*1* K.shape[1], bits=8, bandwidth_ratio=1/head)
                # ######### Preprocessing 
                for k in range(math.ceil((head*ratio*1* K.shape[1])/int(PE_width*PE_height/head))):
                    PRE_cycles += 1
            
            # ############ Q #########
            # ######### Load Q and decoder weights
            reload_ratio = (V.shape[0] - global_tokens)/(my_SRAM.max_K/(head*V.shape[1]*8)-global_tokens)
            reload_ratio = max(reload_ratio, 1)
            if global_tokens==0:
                reload_ratio = len(sparser)/mask[:, global_tokens:].shape[1]
                for i in range(Q.shape[0]):
                    preload_cycles += my_SRAM.preload_Q(nums=head*ratio*1* Q.shape[1], bits=8, bandwidth_ratio=1)*reload_ratio
                    preload_cycles += my_SRAM.preload_decoder(nums=head*ratio*1* Q.shape[1], bits=8, bandwidth_ratio=1/head)*reload_ratio
                    # ######### Preprocessing 
                    for k in range(math.ceil((head*ratio*1* Q.shape[1])/int(PE_width*PE_height/head))):
                        PRE_cycles += 1*reload_ratio
            total_PRE_cycles += PRE_cycles
            total_preload_cycles += preload_cycles
            log.info('Sparse SpMM dataloader | cycles: {}'.format(preload_cycles))
            log.info('Sparse SpMM decoder | cycles: {}'.format(PRE_cycles))
            
            # compuatation
            # K-stationary (Why? Because the number of gloal tokens vary a lot --> Score stationary is not best fit)
            
            # DATA_cycles = 0
            # TODO:
            dense_ratio = global_tokens*Q.shape[0]/(len(sparser) + global_tokens*Q.shape[0])
            dense_PE_width = int(PE_width*dense_ratio)
            sparse_PE_width = PE_width - dense_PE_width
            # ############## dense pattern q*k ##############
            dense_SDDMM_PE_cycles = 0
            for _sta_k in range(global_tokens):
                for _sta_q in range(math.ceil(Q.shape[0]/dense_PE_width)):
                    for _tile_q in range(math.ceil(Q.shape[1] / (PE_height/head))):
                        dense_SDDMM_PE_cycles += 1
            log.info('Dense SDMM PE caclulation | cycles: {}'.format(dense_SDDMM_PE_cycles))
            # ############## simoutalous sparse pattern q*k ##############
            sparse_SDDMM_PE_cycles = 0
            # for _sta_k in range(math.ceil(len(sparser)*Q.shape[1]/int(sparse_PE_width*PE_height/head))):
            for _sta_k in range(math.ceil(len(sparser)/(sparse_PE_width))):
                for _tile_q in range(math.ceil(Q.shape[1] / (PE_height/head))):
                    sparse_SDDMM_PE_cycles += 1
            log.info('Sparse SDMM PE caclulation | cycles: {}'.format(sparse_SDDMM_PE_cycles))
            SDDMM_PE_cycles = max(dense_SDDMM_PE_cycles, sparse_SDDMM_PE_cycles)
            total_SDDMM_PE_cycles += SDDMM_PE_cycles
            
            # ----------------------------------------------------------------------------
              
            log.info('***' * 4)
            attn_map = np.zeros((Q.shape[0], K.shape[0]))
            preload_cycles = 0
            for _tile_k in range(global_tokens):
                preload_cycles += my_SRAM.preload_V(nums=head*1* V.shape[1], bits=8)
            total_preload_cycles += preload_cycles
            log.info('Dense SpMM dataloader | cycles: {}'.format(preload_cycles))
            # ############## dense pattern s*v ##############
            dense_SpMM_PE_cycles = 0
            for _tile_attn in range(math.ceil((V.shape[0]*V.shape[1]*global_tokens) / int(dense_PE_width*PE_height/head))):
                dense_SpMM_PE_cycles += 1
            # total_SpMM_PE_cycles += SpMM_PE_cycles
            log.info('Dense SpMM PE caclulation | cycles: {}'.format(dense_SpMM_PE_cycles))
            
            # ############## sparse pattern s*v ##############
            # acumulation
            row_index = [i for i in range(V.shape[0])]
            num_list = []
            accumulator = 0
            i = 0
            for _q_index, _k_index in sparser:
                if _q_index == row_index[i]:
                    accumulator += 1
                else:
                    if accumulator == 0:
                        pass
                    else:
                        num_list.append(accumulator)
                        accumulator = 0
                    i = _q_index
                    accumulator = 1
            num_list.append(accumulator)

            sparse_SpMM_PE_cycles = 0
            preload_cycles = 0
            for _tile_k in range(attn_map.shape[0]-global_tokens): 
                preload_cycles += my_SRAM.preload_V(nums=head*1* V.shape[1], bits=8)*(1+0.5)
            total_preload_cycles += preload_cycles
            log.info('Sparse SpMM dataloader | cycles: {}'.format(preload_cycles))
            # ############## sparse pattern s*v ##############
            SpMM_PE_cycles = 0
            for row_num in num_list:
                sparse_SpMM_PE_cycles += row_num*V.shape[1]
            sparse_SpMM_PE_cycles = math.ceil(sparse_SpMM_PE_cycles/int(sparse_PE_width*PE_height/head))
            log.info('Sparse SpMM PE caclulation | cycles: {}'.format(sparse_SpMM_PE_cycles))  
            SpMM_PE_cycles = max(sparse_SpMM_PE_cycles, dense_SpMM_PE_cycles)
            total_SpMM_PE_cycles += SpMM_PE_cycles
            
            # for row_num in range(int(len(num_list)/PE_height)):
            #     # for _tile_attn in range(int(row_num / PE_height)):
            #     for _tile_v in range(int(V.shape[1] / PE_width)):  # do not need to plus one if 64 / 64 == 0
            #         for _tile_k in range(num_list[row_num]): 
            #             SpMM_PE_cycles += 1

            # ########### linear transformation
            # Linear_PE_cycles = 0
            # for _tile_attn in range(int(attn_map.shape[0] / PE_height)):
            #     for _tile_v in range(int(V.shape[1] / PE_width)):  
            #         for _tile_k in range(V.shape[0]):
            #             Linear_PE_cycles += 1
            # print('Linear PE caclulation | cycles: {}'.format(Linear_PE_cycles))
            # total_linear_PE_cycles += Linear_PE_cycles
        
        # ############## Postprocessing
        # PRE_cycles = 0
        # store_cycles = 0
        # Q = all_Q[_layer, _head]
        # K = all_K[_layer, _head]
        # V = all_V[_layer, _head]
        # for h in range(6):
        #     for q in range(int((Q.shape[0]*Q.shape[1])/(PE_width*PE_width))):
        #         for h in range(12):
        #             PRE_cycles += 1
            
        #     for k in range(int((K.shape[0]*K.shape[1])/(PE_width*PE_width))):
        #         for h in range(12):
        #             PRE_cycles += 1

        #     store_cycles += my_SRAM.store_out(nums=V.shape[0] * V.shape[1], bits=8)
        #     # Data_cycles.append()
        #     # preload_cycles += my_SRAM.preload_index(nums=len(sparser)*2, bits=8)
        #     print('store output to SRAM | cycles: {}'.format(store_cycles))
        #     total_preload_cycles += store_cycles

        # total_PRE_cycles += PRE_cycles
        # print("Postprocess cycles: {}". format(PRE_cycles))
    
    log.info('')
    log.info('***' * 10)
    log.info('number of non-zeros in the sparser region: {}'.format(total_sparse_ratio/all_Q.shape[0]))
    log.info('total preloading cycles: {}'.format(total_preload_cycles))
    log.info('total processing cycles: {}'.format(total_PRE_cycles))
    log.info('total Computation cycles: {}'.format(total_SDDMM_PE_cycles+total_SpMM_PE_cycles))

    log.info('')
    log.info('***' * 10)
    log.info('Total cycles: {}'.format(max(total_preload_cycles, total_PRE_cycles+total_SDDMM_PE_cycles+total_SpMM_PE_cycles)))
