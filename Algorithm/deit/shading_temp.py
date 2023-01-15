import matplotlib.pyplot as plt
import os, sys
import numpy as np
import re
import matplotlib.ticker as ticker

def mySaveFig(pltm,fntmp,fp=0,isax=0,iseps=0,isShowPic=0):
    # save config
    font = {'weight': 'bold'}
    Leftp=0.18
    Bottomp=0.18
    Widthp=0.88-Leftp
    Heightp=0.9-Bottomp
    pos=[Leftp,Bottomp,Widthp,Heightp]

    if isax==1:
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        pltm.yticks(size=14)
        pltm.xticks(size=14)
    fnm='%s.svg'%(fntmp)
    pltm.savefig(fnm, bbox_inches='tight')
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        pltm.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic:
        pltm.show()
    else:
        pltm.close()

def get_mean_std(txts):
    res = np.zeros((5,11,6,3)) #[groups, epochs, g_ratio, w_ratio]
    epoch_list = [0,2,4,6,8,10,20,40,60,80,100]
    gr_list = [0,10,30,50,70,90] # g_ratio: 0,10,30,50,70,90
    for group in range(0,5):
        for k in range(0,11):# for 0,2,4,6,8,10,20,40,60,80,100 epochs
            epoch = epoch_list[k]
            f = open(txts[group],'r')
            cnt = 1
            line = f.readline()
            line = line[:-1]
            for head in range(0,k*18):
                cnt += 1
                line = f.readline()
                line = line[:-1]
            # print('epoch = %d, cnt start = %d'%(epoch,cnt))
            offset = k*18
            g_ratio = int((cnt-offset-1)/3)
            w_ratio = int((cnt-offset-1)%3)
            if(cnt-offset<=18):
                if(line[26]=='T'):
                    acc_str = line[32:38]
                    acc_f = float(acc_str)
                    res[group][k][g_ratio][w_ratio] = acc_f
                    # res_acc_sum[k] += acc_f
            while line:
                cnt += 1
                line = f.readline()
                line = line[:-1]
                g_ratio = int((cnt-offset-1)/3)
                w_ratio = int((cnt-offset-1)%3)
                if(cnt-offset<=18):
                    if(line[26]=='T'):
                        acc_str = line[32:38]
                        acc_f = float(acc_str)
                        res[group][k][g_ratio][w_ratio] = acc_f
                        # res_acc_sum[k] += acc_f
            # print('cnt end = %d'%(cnt))
            f.close()
    res *= 100
    # for group in range(5):
    #     print('group = ',group+1)
    #     for k in range(0,11):
    #         print(res[group][k])
    #         # epoch = epoch_list[k]
    #         # print('epoch = %d, acc_avg = %.4f'%(epoch,acc_avg[k]))
    #         print('\n')


    res = np.swapaxes(res, 1, 3)# [groups, w_ratio, g_ratio, epochs] (5,3,6,11)

    res_mean = np.mean(res,axis=0) # (3,6,11)
    res_err = np.std(res,axis=0) # (3,6,11)

    return res_mean, res_err


# load results
dataset = 'CiteSeer'
folder = 'jointEB_warmup_25_CiteSeer/'
# txts = ['test_joint_EB_warmup_g1_no_retrain_CiteSeer.txt','test_joint_EB_warmup_g2_no_retrain_CiteSeer.txt','test_joint_EB_warmup_g3_no_retrain_CiteSeer.txt','test_joint_EB_warmup_g4_no_retrain_CiteSeer.txt','test_joint_EB_warmup_g5_no_retrain_CiteSeer.txt']
txts = ['test_joint_EB_warmup_g1_CiteSeer.txt','test_joint_EB_warmup_g2_CiteSeer.txt','test_joint_EB_warmup_g3_CiteSeer.txt','test_joint_EB_warmup_g4_CiteSeer.txt','test_joint_EB_warmup_g5_CiteSeer.txt']
for i in range(len(txts)):
    txts[i] = folder + txts[i]
meanArr_CiteSeer, stdArr_CiteSeer = get_mean_std(txts)

dataset = 'Cora'
folder = 'jointEB_warmup_25_Cora/'
txts = ['test_joint_EB_warmup_g1_Cora.txt','test_joint_EB_warmup_g2_Cora.txt','test_joint_EB_warmup_g3_Cora.txt','test_joint_EB_warmup_g4_Cora.txt','test_joint_EB_warmup_g5_Cora.txt']
for i in range(len(txts)):
    txts[i] = folder + txts[i]
meanArr_Cora, stdArr_Cora = get_mean_std(txts)


x = [0,2,4,6,8,10,20,40,60,80,100]
length = len(x) # 11
base_value_0 = meanArr_CiteSeer[0][0].mean()
base_value_1 = meanArr_CiteSeer[1][0].mean()
base_value_2 = meanArr_CiteSeer[2][0].mean()
base_value_CiteSeer = np.mean([base_value_0,base_value_1,base_value_2])
print("base_value_CiteSeer = ",base_value_CiteSeer)
unpruned_baseline_CiteSeer = [base_value_CiteSeer for i in range(length)]

base_value_0 = meanArr_Cora[0][0].mean()
base_value_1 = meanArr_Cora[1][0].mean()
base_value_2 = meanArr_Cora[2][0].mean()
base_value_Cora = np.mean([base_value_0,base_value_1,base_value_2])
print("base_value_Cora = ",base_value_Cora)
unpruned_baseline_Cora = [base_value_Cora for i in range(length)]
# SGCN_citeseer_baseline = [78.41 for i in range(length)]


colors = ['purple','g','b','r','c']
marker = ['^', 's', 'p', 'o', '*'] # ['.', 's', '>', '8', 'h']
font_big = 54
font_mid = 44
font_leg = 33
font_small = 30
lw = 8

x_axis = [0,2,4,6,8,10,20,40,60,80,100]
length = len(x_axis)
x_range = [i for i in range(len(x_axis))]

ylim_min_CiteSeer = [53, 53, 53]
ylim_max_CiteSeer = [73, 73, 73]
ylim_min_Cora = [50, 50, 50]
ylim_max_Cora = [85, 85, 85]

########################################################

# fig, ax = plt.subplots(2, 3, figsize=(48, 25))
# # fig.text(0.430, -0.13, "CiteSeer With Retrain", fontsize=font_big, fontweight="bold", color="k")
# # fig.text(0.675, -0.13, "CIFAR-100", fontsize=font_big, fontweight="bold", color="k")
# plt.subplots_adjust(wspace =0.1, hspace =0.2)

# for i in range(3):
#     ax[0][i].plot(x_range, unpruned_baseline_Cora, 'k--',  lw=4, label=r'$\bf{Unpruned}$')
#     ax[0][i].errorbar(x_range, meanArr_Cora[i][1], fmt=colors[0], marker=marker[0], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 10\%}$', yerr=stdArr_Cora[i][1])
#     ax[0][i].errorbar(x_range, meanArr_Cora[i][2], fmt=colors[1], marker=marker[1], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 30\%}$', yerr=stdArr_Cora[i][2])
#     ax[0][i].errorbar(x_range, meanArr_Cora[i][3], fmt=colors[2], marker=marker[2], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 50\%}$', yerr=stdArr_Cora[i][3])
#     ax[0][i].errorbar(x_range, meanArr_Cora[i][4], fmt=colors[3], marker=marker[3], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 70\%}$', yerr=stdArr_Cora[i][4])
#     ax[0][i].errorbar(x_range, meanArr_Cora[i][5], fmt=colors[4], marker=marker[4], markersize=4*lw, lw=lw, label=r'$\bf{p_g = 90\%}$', yerr=stdArr_Cora[i][5])
#     ax[0][i].set_ylim([ylim_min_Cora[i], ylim_max_Cora[i]])
#     ax[0][i].xaxis.set_tick_params(labelsize=font_small)
#     ax[0][i].yaxis.set_tick_params(labelsize=font_small)
#     ax[0][i].set_xticks(x_range)
#     ax[0][i].set_xticklabels(x)
#     # if i == 1:
#     #     ax[0][i].set_xlabel('Epoch subgraphs and subnetworks drawn from', fontsize=font_big, fontweight="bold", y=0.98)
#     ax[0][i].get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%d'))

#     ax[1][i].plot(x_range, unpruned_baseline_CiteSeer, 'k--',  lw=4, label=r'$\bf{Unpruned}$')
#     ax[1][i].errorbar(x_range, meanArr_CiteSeer[i][1], fmt=colors[0], marker=marker[0], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 10\%}$', yerr=stdArr_CiteSeer[i][1])
#     ax[1][i].errorbar(x_range, meanArr_CiteSeer[i][2], fmt=colors[1], marker=marker[1], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 30\%}$', yerr=stdArr_CiteSeer[i][2])
#     ax[1][i].errorbar(x_range, meanArr_CiteSeer[i][3], fmt=colors[2], marker=marker[2], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 50\%}$', yerr=stdArr_CiteSeer[i][3])
#     ax[1][i].errorbar(x_range, meanArr_CiteSeer[i][4], fmt=colors[3], marker=marker[3], markersize=3*lw, lw=lw, label=r'$\bf{p_g = 70\%}$', yerr=stdArr_CiteSeer[i][4])
#     ax[1][i].errorbar(x_range, meanArr_CiteSeer[i][5], fmt=colors[4], marker=marker[4], markersize=4*lw, lw=lw, label=r'$\bf{p_g = 90\%}$', yerr=stdArr_CiteSeer[i][5])
#     ax[1][i].set_ylim([ylim_min_CiteSeer[i], ylim_max_CiteSeer[i]])
#     ax[1][i].xaxis.set_tick_params(labelsize=font_small)
#     ax[1][i].yaxis.set_tick_params(labelsize=font_small)
#     ax[1][i].set_xticks(x_range)
#     ax[1][i].set_xticklabels(x)
#     if i == 1:
#         ax[1][i].set_xlabel('Epoch subgraphs and subnetworks drawn from', fontsize=font_big, fontweight="bold", y=0.98)
#     ax[1][i].get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%d'))

# # ax[0][0].set_ylabel('Retraining Accuracy (%)', fontsize=font_big, fontweight="bold", x=0.5)
# # ax[1][0].set_ylabel('Retraining Accuracy (%)', fontsize=font_big, fontweight="bold")
# fig.text(0.1, 0.5, 'Retraining accuracy (%)', fontsize=font_big+5, fontweight="bold", ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.03, '(a) The existence of Joint-EB tickets', fontsize=font_big+5, fontweight="bold", ha='center', va='center')
# ax[0][0].set_title(r'$\bf{Cora\ (p_w = 50\%)}$', fontsize=font_big, fontweight="bold", y=1.02)
# ax[0][1].set_title(r'$\bf{Cora\ (p_w = 70\%)}$', fontsize=font_big, fontweight="bold", y=1.02)
# ax[0][2].set_title(r'$\bf{Cora\ (p_w = 90\%)}$', fontsize=font_big, fontweight="bold", y=1.02)
# ax[1][0].set_title(r'$\bf{CiteSeer\ (p_w = 50\%)}$', fontsize=font_big, fontweight="bold", y=1.02)
# ax[1][1].set_title(r'$\bf{CiteSeer\ (p_w = 70\%)}$', fontsize=font_big, fontweight="bold", y=1.02)
# ax[1][2].set_title(r'$\bf{CiteSeer\ (p_w = 90\%)}$', fontsize=font_big, fontweight="bold", y=1.02)

# leg1 = ax[0][0].legend(fontsize=font_leg, ncol=2, loc='lower right')
# leg1.get_frame().set_edgecolor("black")
# leg1.get_frame().set_linewidth(5)
# leg2 = ax[0][1].legend(fontsize=font_leg, ncol=2, loc='lower right')
# leg2.get_frame().set_edgecolor("black")
# leg2.get_frame().set_linewidth(5)
# leg3 = ax[0][2].legend(fontsize=font_leg, ncol=2, loc='lower right')
# leg3.get_frame().set_edgecolor("black")
# leg3.get_frame().set_linewidth(5)

# leg1 = ax[1][0].legend(fontsize=font_leg, ncol=2, loc='lower right')
# leg1.get_frame().set_edgecolor("black")
# leg1.get_frame().set_linewidth(5)
# leg2 = ax[1][1].legend(fontsize=font_leg, ncol=2, loc='lower right')
# leg2.get_frame().set_edgecolor("black")
# leg2.get_frame().set_linewidth(5)
# leg3 = ax[1][2].legend(fontsize=font_leg, ncol=2, loc='lower right')
# leg3.get_frame().set_edgecolor("black")
# leg3.get_frame().set_linewidth(5)

# for i in range(2):
#     for j in range(3):
#         ax[i][j].spines['bottom'].set_linewidth(6)
#         ax[i][j].spines['bottom'].set_color('black')
#         ax[i][j].spines['left'].set_linewidth(6)
#         ax[i][j].spines['left'].set_color('black')
#         ax[i][j].spines['top'].set_linewidth(6)
#         ax[i][j].spines['top'].set_color('black')
#         ax[i][j].spines['right'].set_linewidth(6)
#         ax[i][j].spines['right'].set_color('black')

# mySaveFig(plt, 'Joint_EB', isax=0, fp=1, isShowPic=0)


########################################################
def norm_dw(y):
    max_value = max(y)
    min_value = min(y)
    for i, item in enumerate(y):
        y[i] = (item - min_value) / (max_value - min_value)
    return y

def norm_dg(y):
    max_value = max(y)
    min_value = min(y)
    for i, item in enumerate(y):
        y[i] = item / max_value
    return y

def get_dist_dg(logfiles):
    x = [i for i in range(3, 100)]
    dist = np.zeros((12, len(x)))
    i = 0
    for logfile in logfiles:
        f = open(logfile, 'r')
        line = f.readline()
        line = line.strip()
        dist[i][0] = line
        cnt = 0
        while line and len(line.strip()) != 0:
            dist[i][cnt] = line.strip()
            cnt += 1
            line = f.readline()
        i += 1
    new_dist = []
    for i in range(12):
        new_dist.append(list(norm_dg(dist[i])))
    print(len(new_dist), len(new_dist[0])) # 12 x 97
    return new_dist

def get_dist_dw(logfiles):
    x = [i for i in range(3, 100)]
    dist = np.zeros((12, len(x)))
    i = 0
    for logfile in logfiles:
        f = open(logfile, 'r')
        line = f.readline()
        line = line.strip()
        dist[i][0] = line
        cnt = 0
        while line and len(line.strip()) != 0:
            dist[i][cnt] = line.strip()
            cnt += 1
            line = f.readline()
        i += 1
    new_dist = []
    for i in range(12):
        new_dist.append(list(norm_dg(dist[i])))
    print(len(new_dist), len(new_dist[0])) # 12 x 97
    return new_dist

def get_dist_dg_dw(logfiles):
    x = [i for i in range(3, 100)]
    dist_dw = []
    dist_dg = []
    i = 0

    for logfile in logfiles:
        y_weight = np.zeros((100))
        y_graph = np.zeros((100))
        y_joint = np.zeros((100))

        f = open(logfile, 'r')
        cnt = 1
        cnt -= 1
        if 'Cora' in logfile:
            a = f.readline()
            a = f.readline()
        line = f.readline()
        line = line[:-1]
        g_ratio = int((cnt-1)/4)
        w_ratio = int((cnt-1)%4)
        if(line[44]=='j'):
            weight_dist_str = line[21:26]
            graph_dist_str = line[37:42]
            acc_str = line[57:62]
            weight_dist_f = float(weight_dist_str)
            graph_dist_f = float(graph_dist_str)
            acc_f = float(acc_str)

            y_weight[cnt-1] = weight_dist_f
            y_graph[cnt-1] = graph_dist_f
            y_joint[cnt-1] = acc_f
        while line:
            cnt += 1
            line = f.readline()
            line = line[:-1]
            if(len(line)<2):
                break
            g_ratio = int((cnt-1)/4)
            w_ratio = int((cnt-1)%4)
            if(line[44]=='j'):
                weight_dist_str = line[21:26]
                graph_dist_str = line[37:42]
                acc_str = line[57:62]
                weight_dist_f = float(weight_dist_str)
                graph_dist_f = float(graph_dist_str)
                acc_f = float(acc_str)

                y_weight[cnt-1] = weight_dist_f
                y_graph[cnt-1] = graph_dist_f
                y_joint[cnt-1] = acc_f
        f.close()
        dist_dw.append(y_weight[3:])
        dist_dg.append(y_graph[3:])

    new_dist_dg = []
    new_dist_dw = []
    for i in range(12):
        new_dist_dg.append(list(norm_dg(dist_dg[i])))
        new_dist_dw.append(list(norm_dg(dist_dw[i])))

    print(len(dist_dw), len(dist_dw[0])) # 12 x 97
    return new_dist_dg, new_dist_dw

folder = 'jointEB_points_Cora/'
logfiles = []
for g_ratio in [20, 40, 60, 80]:
    for w_ratio in [50, 70, 90]:
        logfiles.append(folder + 'jointEB_Gr{}_Wr{}_dtG_Cora.txt'.format(str(g_ratio), str(w_ratio)))
Cora_dg = get_dist_dg(logfiles)
logfiles = []
for g_ratio in [20, 40, 60, 80]:
    for w_ratio in [50, 70, 90]:
        logfiles.append(folder + 'jointEB_Gr{}_Wr{}_dtW_Cora.txt'.format(str(g_ratio), str(w_ratio)))
Cora_dw = get_dist_dw(logfiles)
Cora_dist = []
for i in range(12):
    dg_list = Cora_dg[i]
    dw_list = Cora_dw[i]
    d = []
    for j in range(len(dg_list)):
        d.append((dg_list[j] + dw_list[j]) / 2)
    Cora_dist.append(d)
print(len(Cora_dist), len(Cora_dist[0]))

folder = 'jointEB_points_CiteSeer/'
logfiles = []
for g_ratio in [20, 40, 60, 80]:
    for w_ratio in [50, 70, 90]:
        logfiles.append(folder + 'jointEB_Gr{}_Wr{}_dtG_CiteSeer.txt'.format(str(g_ratio), str(w_ratio)))
CiteSeer_dg = get_dist_dg(logfiles)
logfiles = []
for g_ratio in [20, 40, 60, 80]:
    for w_ratio in [50, 70, 90]:
        logfiles.append(folder + 'jointEB_Gr{}_Wr{}_dtW_CiteSeer.txt'.format(str(g_ratio), str(w_ratio)))
CiteSeer_dw = get_dist_dw(logfiles)


folder = 'cite_core/v1/'
logfiles = []
for g_ratio in [20, 40, 60, 80]:
    for w_ratio in [50, 70, 90]:
        logfiles.append(folder + 'Cora_g{}_w{}.txt'.format(str(g_ratio), str(w_ratio)))
Cora_dg, Cora_dw = get_dist_dg_dw(logfiles)
Cora_dist_2 = []
for i in range(12):
    dg_list = Cora_dg[i]
    dw_list = Cora_dw[i]
    d = []
    for j in range(len(dg_list)):
        d.append((dg_list[j] + dw_list[j]) / 2)
    Cora_dist_2.append(d)
print(len(Cora_dist_2), len(Cora_dist_2[0]))

folder = 'citeseer_traj/v2/'
logfiles = []
for g_ratio in [20, 40, 60, 80]:
    for w_ratio in [50, 70, 90]:
        logfiles.append(folder + 'CiteSeer_g{}_w{}.txt'.format(str(g_ratio), str(w_ratio)))
CiteSeer_dg, CiteSeer_dw = get_dist_dg_dw(logfiles)

CiteSeer_dist = []
for i in range(12):
    dg_list = CiteSeer_dg[i]
    dw_list = CiteSeer_dw[i]
    d = []
    for j in range(len(dg_list)):
        d.append((dg_list[j] + dw_list[j]) / 2)
    CiteSeer_dist.append(d)
print(len(CiteSeer_dist), len(CiteSeer_dist[0]))

# # graph distance + network distance
x_range = [i for i in range(3,100)]
threshold = 0.1


# #graph ratio:20 weight ratio:50 epoch:3-99
# x0 = [i+3 for i in range(97)]
# y0 = [37280, 36084, 29636, 27315, 27720, 25525, 17192, 18134, 19262, 19958, 17346, 18367, 17452, 16324, 15820, 14561, 14053, 12926, 14985, 15276, 19335, 13639, 19151, 19535, 17012, 15283, 16807, 14603, 16949, 13494, 12016, 18892, 15322, 13280, 19331, 11879, 13989, 14008, 21540, 22050, 21166, 21592, 14466, 17918, 15048, 13504, 17896, 15212, 16726, 22234, 19690, 19600, 13902, 19788, 20108, 14211, 16233, 11781, 11470, 14478, 20408, 22356, 22060, 20382, 15836, 22377, 13782, 22759, 17026, 12064, 20020, 16554, 18588, 18435, 16330, 19192, 20242, 19038, 16814, 21480, 19486, 17224, 18960, 15674, 19348, 15046, 14932, 13574, 20536, 15333, 17275, 16161, 17011, 17027, 15253, 17288, 17583]
# y0 = norm(y0)
# #graph ratio:20 weight ratio:70 epoch:3-99
# x1 = [i+3 for i in range(97)]
# y1 = [27215, 24282, 19416, 13716, 14543, 15310, 13177, 13786, 12841, 13297, 16218, 13292, 13490, 12660, 13326, 10731, 10463, 11584, 12328, 11488, 12075, 9398, 8914, 8386, 10047, 14313, 12029, 10194, 14280, 9675, 11071, 9644, 10577, 11718, 12016, 9532, 8818, 9436, 9417, 12223, 12433, 9683, 12098, 6850, 7306, 8386, 6877, 8688, 9738, 9150, 7004, 9587, 10840, 8090, 12354, 9752, 9852, 10223, 11656, 14438, 13044, 10391, 9754, 9488, 9658, 10654, 8453, 6830, 9212, 10752, 9222, 13274, 10699, 11024, 11182, 11089, 13382, 10639, 12467, 13284, 9194, 9302, 11592, 10688, 11706, 8923, 12313, 11060, 10118, 13518, 12863, 16372, 12096, 14682, 12401, 12984, 16902]
# y1 = norm(y1)
# #graph ratio:20 weight ratio:90 epoch:3-99 unfinished: only 85 points
# x2 = [i+3 for i in range(85)]
# y2 = [9979, 7055, 6165, 5362, 5817, 5872, 5138, 5052, 4535, 6274, 4866, 5254, 5113, 4184, 6059, 5168, 4832, 5098, 4420, 4802, 5402, 4068, 4384, 4750, 4130, 3188, 4251, 3756, 3652, 2834, 2796, 3049, 3327, 3244, 3305, 2956, 2487, 3292, 2954, 3342, 2942, 2512, 2885, 3234, 3088, 3962, 3419, 4128, 3919, 4695, 4436, 3682, 4004, 3777, 4592, 3469, 3720, 4370, 2947, 4144, 3119, 3124, 3526, 3070, 3032, 3960, 2886, 4000, 3610, 2810, 4836, 3978, 5050, 4441, 3550, 3752, 3892, 3312, 5142, 4720, 4924, 5388, 3056, 2656, 2534]
# y2 = norm(y2)

x_range = [i+3 for i in range(97)]

fig, ax = plt.subplots(2, 1, figsize=(16.5, 25))
plt.subplots_adjust(wspace =0.1, hspace =0.2)

ax[0].plot(x_range, Cora_dist[0], c='purple', lw=lw, label=r'$\bf{p_g = 20\%;\ p_w = 50\%}$')
ax[0].plot(x_range, Cora_dist[1], c='g', lw=lw, label=r'$\bf{p_g = 20\%;\ p_w = 70\%}$')
ax[0].plot(x_range, Cora_dist[2], c='b', lw=lw, label=r'$\bf{p_g = 20\%;\ p_w = 90\%}$')
ax[0].plot(x_range, Cora_dist[3], c='c', lw=lw, label=r'$\bf{p_g = 40\%;\ p_w = 50\%}$')
ax[0].plot(x_range, Cora_dist[4], c='m', lw=lw, label=r'$\bf{p_g = 40\%;\ p_w = 70\%}$')
ax[0].plot(x_range, Cora_dist[5], c='y', lw=lw, label=r'$\bf{p_g = 40\%;\ p_w = 90\%}$')
ax[0].plot(x_range, Cora_dist[6], c='k', lw=lw, label=r'$\bf{p_g = 60\%;\ p_w = 50\%}$')
ax[0].plot(x_range, Cora_dist[7], c='pink', lw=lw, label=r'$\bf{p_g = 60\%;\ p_w = 70\%}$')
ax[0].plot(x_range, Cora_dist_2[8], c='chocolate', lw=lw, label=r'$\bf{p_g = 60\%;\ p_w = 90\%}$')
ax[0].plot(x_range, Cora_dist[9], c='darkviolet', lw=lw, label=r'$\bf{p_g = 80\%;\ p_w = 50\%}$')
ax[0].plot(x_range, Cora_dist[10], c='orange', lw=lw, label=r'$\bf{p_g = 80\%;\ p_w = 70\%}$')
ax[0].plot(x_range, Cora_dist[11], c='royalblue', lw=lw, label=r'$\bf{p_g = 80\%;\ p_w = 90\%}$')

ax[0].set_ylim([0, 1])
ax[0].xaxis.set_tick_params(labelsize=font_small)
ax[0].yaxis.set_tick_params(labelsize=font_small)
ax[0].set_title('Cora', fontsize=font_big, fontweight="bold")
ax[0].plot(x_range, [threshold]*97, c='r', lw=4, alpha=0.7)
print(np.where((np.array(Cora_dist[-4]) >= threshold-0.01) & (np.array(Cora_dist[-8]) <= threshold+0.01) & (np.array(x_range) < 50)), 'should + 3!!!')
ax[0].fill_between(x_range, 0, 1, where=(np.array(Cora_dist[-4]) >= threshold-0.01) & (np.array(Cora_dist[-8]) <= threshold+0.01) & (np.array(x_range) < 50),
    color='green', alpha=0.2, transform=ax[0].get_xaxis_transform())
fig.text(0.48, 0.577, r'$\bf{threshold\ \eta=0.1}$', c='r', alpha=0.7, fontsize=font_leg, fontweight="bold", ha='center', va='center')
fig.text(0.333, 0.75, r'Joint-EB tickets', fontsize=font_leg, fontweight="bold", ha='left', va='center')
fig.text(0.333, 0.73, r'emerge at:', fontsize=font_leg, fontweight="bold", ha='left', va='center')
fig.text(0.333, 0.70, '9 - 26 epochs', c='green', alpha=0.7, fontsize=font_leg, fontweight="bold", ha='left', va='center')
ax[0].set_xlim([0,99])
ax[0].grid()


ax[1].plot(x_range, CiteSeer_dist[0], c='purple', lw=lw, label=r'$\bf{p_g = 20\%;\ p_w = 50\%}$')
ax[1].plot(x_range, CiteSeer_dist[1], c='g', lw=lw, label=r'$\bf{p_g = 20\%;\ p_w = 70\%}$')
ax[1].plot(x_range, CiteSeer_dist[2], c='b', lw=lw, label=r'$\bf{p_g = 20\%;\ p_w = 90\%}$')
ax[1].plot(x_range, CiteSeer_dist[3], c='c', lw=lw, label=r'$\bf{p_g = 40\%;\ p_w = 50\%}$')
ax[1].plot(x_range, CiteSeer_dist[4], c='m', lw=lw, label=r'$\bf{p_g = 40\%;\ p_w = 70\%}$')
ax[1].plot(x_range, CiteSeer_dist[5], c='y', lw=lw, label=r'$\bf{p_g = 40\%;\ p_w = 90\%}$')
ax[1].plot(x_range, CiteSeer_dist[6], c='k', lw=lw, label=r'$\bf{p_g = 60\%;\ p_w = 50\%}$')
ax[1].plot(x_range, CiteSeer_dist[7], c='pink', lw=lw, label=r'$\bf{p_g = 60\%;\ p_w = 70\%}$')
ax[1].plot(x_range, CiteSeer_dist[8], c='chocolate', lw=lw, label=r'$\bf{p_g = 60\%;\ p_w = 90\%}$')
# ax[1].plot(x_range, CiteSeer_dist[9], c='darkviolet', lw=lw, label=r'$\bf{p_g = 80\%;\ p_w = 50\%}$')
ax[1].plot(x_range, CiteSeer_dist[10], c='orange', lw=lw, label=r'$\bf{p_g = 80\%;\ p_w = 70\%}$')
ax[1].plot(x_range, CiteSeer_dist[11], c='royalblue', lw=lw, label=r'$\bf{p_g = 80\%;\ p_w = 90\%}$')

ax[1].set_ylim([0, 1])
ax[1].xaxis.set_tick_params(labelsize=font_small)
ax[1].yaxis.set_tick_params(labelsize=font_small)
ax[1].set_title('CiteSeer', fontsize=font_big, fontweight="bold")
ax[1].set_xlabel('Epochs', fontsize=font_big, fontweight="bold")
ax[1].plot(x_range, [threshold]*97, c='r', lw=4, alpha=0.7)
print(np.where((np.array(CiteSeer_dist[1]) >= threshold-0.01) & (np.array(CiteSeer_dist[-4]) <= threshold+0.01) & (np.array(x_range) < 50)), 'should + 3!!!')
ax[1].fill_between(x_range, 0, 1, where=(np.array(CiteSeer_dist[1]) >= threshold-0.01) & (np.array(CiteSeer_dist[-4]) <= threshold+0.03) & (np.array(x_range) < 50),
    color='green', alpha=0.2, transform=ax[1].get_xaxis_transform())
fig.text(0.48, 0.157, r'$\bf{threshold\ \eta=0.1}$', c='r', alpha=0.7, fontsize=font_leg, fontweight="bold", ha='center', va='center')
fig.text(0.333, 0.33, r'Joint-EB tickets', fontsize=font_leg, fontweight="bold", ha='left', va='center')
fig.text(0.333, 0.31, r'emerge at:', fontsize=font_leg, fontweight="bold", ha='left', va='center')
fig.text(0.333, 0.28, '13 - 22 epochs', c='green', alpha=0.7, fontsize=font_leg, fontweight="bold", ha='left', va='center')
ax[1].set_xlim([0,99])
ax[1].grid()

fig.text(0.04, 0.5, r'$\bf{Nomarlized\ Distance\ (d_g + d_w)}$', fontsize=font_big+5, fontweight="bold", ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.03, '(b) Distance trajectory', fontsize=font_big+5, fontweight="bold", ha='center', va='center')

leg1 = ax[0].legend(fontsize=font_leg-8, ncol=1, loc='upper right')
leg1.get_frame().set_edgecolor("black")
leg1.get_frame().set_linewidth(5)
leg2 = ax[1].legend(fontsize=font_leg-8, ncol=1, loc='upper right')
leg2.get_frame().set_edgecolor("black")
leg2.get_frame().set_linewidth(5)

for i in range(2):
    ax[i].spines['bottom'].set_linewidth(6)
    ax[i].spines['bottom'].set_color('black')
    ax[i].spines['left'].set_linewidth(6)
    ax[i].spines['left'].set_color('black')
    ax[i].spines['top'].set_linewidth(6)
    ax[i].spines['top'].set_color('black')
    ax[i].spines['right'].set_linewidth(6)
    ax[i].spines['right'].set_color('black')

mySaveFig(plt, 'Joint_EB_Distance', isax=0, fp=1, isShowPic=0)


########################################################


