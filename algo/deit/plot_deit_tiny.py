import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

  
def parse(d):
    dictionary = dict()
    pairs = d.strip('{}').split(', ')
    for i in pairs:
        pair = i.split(': ')
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')
    return dictionary

def get_data(filename):
    acc = []
    loss = []
    rec_loss = []
    try:
        data = open(filename, 'rt')
        lines = data.read().split('\n')
        for l in lines:
            if l != '':
                dictionary = parse(l)
                acc.append(dictionary['test_acc1'])
                loss.append(dictionary['test_loss'])
                rec_loss.append(dictionary['train_recon_loss'])
        data.close()
    except:
        print("Error")
    return [float(x) for x in acc], [float(x) for x in loss], [float(x) for x in rec_loss]


def extend_data(acc, loss, recon_loss, epochs=100):
    if len(acc) >= epochs:
        return acc[:epochs], loss[:epochs], recon_loss[:epochs]

    acc = acc + [None] * (epochs - len(acc))
    loss = loss + [None] * (epochs - len(loss))
    recon_loss = recon_loss + [None] * (epochs - len(recon_loss))

    return acc, loss, recon_loss

def get_recon_loss(filename):
    recon_loss = []
    try:
        data = open(filename, 'rt')
        lines = data.read().split('\n')
        for l in lines:
            recon_loss.append(l.split('recon_loss: ')[1].split(' ')[0])
        data.close()
    except:
        print('Error')
    return [float(x) * 1e4 for x in recon_loss]

# deit_small

# acc0, loss0 =  (get_data('./exp_hr/info_0.9/deit_small_5e-5_100/log.txt'))
# acc1, loss1 =  (get_data('./exp_hr/info_0.9/deit_small_1e-5_100/log.txt'))
# acc2, loss2 =  (get_data('./exp_hr/info_0.9/deit_small_5e-6_100/log.txt'))
# acc3, loss3 =  (get_data('./exp_hr/info_0.9/deit_small_1e-6_100/log.txt'))
# acc4, loss4 =  (get_data('./exp_hr/info_0.9/deit_small_1e-5_5e-6_100/log.txt'))
# acc5, loss5 =  (get_data('./exp_hr/info_0.9/deit_small_1e-5_100_resume/log.txt'))

# acc0, loss0 = extend_data(acc0, loss0)
# acc1, loss1 = extend_data(acc1, loss1)
# acc2, loss2 = extend_data(acc2, loss2)
# acc3, loss3 = extend_data(acc3, loss3)
# acc4, loss4 = extend_data(acc4, loss4)
# acc5, loss5 = extend_data(acc5, loss5)

# f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
# x = range(100)
# ax1.plot(x, acc0, label = "lr=5e-5")
# ax1.plot(x, acc1, label = "lr=1e-5")
# ax1.plot(x, acc2, label = "lr=5e-6")
# ax1.plot(x, acc3, label = "lr=1e-6")
# ax1.plot(x, acc4, label = "lr=1e-5_5e-6")
# ax1.plot(x, acc5, label = "lr=1e-5 (resume)")
# ax1.plot(x[:50], [79.9] * 50, '--', label = 'unpruned')
# ax1.title.set_text('Accuracy')
# ax2.plot(x, loss0, label = "lr=5e-5")
# ax2.plot(x, loss1, label = "lr=1e-5")
# ax2.plot(x, loss2, label = "lr=5e-6")
# ax2.plot(x, loss3, label = "lr=1e-6")
# ax2.plot(x, loss4, label = "lr=1e-5_5e-6")
# ax2.plot(x, loss5, label = "lr=1e-5 (resume)")
# ax2.title.set_text('Test loss')
# # handles, labels = ax.get_legend_handles_labels()
# # fig.legend(handles, labels, loc='upper center')
# # ax1.plot(rng, line3, label = "sparsity=90.36 lr=1e-5")
# plt.legend()
# f.tight_layout()
# plt.savefig("./exp_hr/info_0.9_deit_small.png")


# # deit_base

# # acc0, loss0 =  (get_data('./exp_hr/info_0.9/deit_base_5e-5_100/log.txt'))
# acc1, loss1 =  (get_data('./exp_hr/info_0.9/deit_base_1e-5_100/log.txt'))
# acc2, loss2 =  (get_data('./exp_hr/info_0.9/deit_base_5e-6_100/log.txt'))
# # acc3, loss3 =  (get_data('./exp_hr/info_0.9/deit_base_1e-6_100/log.txt'))
# acc4, loss4 =  (get_data('./exp_hr/info_0.9/deit_base_1e-5_5e-6_100/log.txt'))
# acc5, loss5 =  (get_data('./exp_hr/info_0.9/deit_base_1e-5_100_resume/log.txt'))

# # acc0, loss0 = extend_data(acc0, loss0)
# acc1, loss1 = extend_data(acc1, loss1)
# acc2, loss2 = extend_data(acc2, loss2)
# # acc3, loss3 = extend_data(acc3, loss3)
# acc4, loss4 = extend_data(acc4, loss4)
# acc5, loss5 = extend_data(acc5, loss5)

# f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
# x = range(100)
# # ax1.plot(x, acc0, label = "lr=5e-5")
# ax1.plot(x, acc1, label = "lr=1e-5")
# ax1.plot(x, acc2, label = "lr=5e-6")
# # ax1.plot(x, acc3, label = "lr=1e-6")
# ax1.plot(x, acc4, label = "lr=1e-5_5e-6")
# ax1.plot(x, acc5, label = "lr=1e-5 (resume)")
# ax1.plot(x[:50], [81.8] * 50, '--', label = 'unpruned')
# ax1.title.set_text('Accuracy')
# # ax2.plot(x, loss0, label = "lr=5e-5")
# ax2.plot(x, loss1, label = "lr=1e-5")
# ax2.plot(x, loss2, label = "lr=5e-6")
# # ax2.plot(x, loss3, label = "lr=1e-6")
# ax2.plot(x, loss4, label = "lr=1e-5_5e-6")
# ax2.plot(x, loss5, label = "lr=1e-5 (resume)")
# ax2.title.set_text('Test loss')
# # handles, labels = ax.get_legend_handles_labels()
# # fig.legend(handles, labels, loc='upper center')
# # ax1.plot(rng, line3, label = "sparsity=90.36 lr=1e-5")
# plt.legend()
# f.tight_layout()
# plt.savefig("./exp_hr/info_0.9_deit_base.png")


######################### deit_tiny (low rank qk) #########################

# acc0, loss0, recon_loss0 =  (get_data('exp_hr/svd/deit_tiny_1e-4_5e-6_100_mix_head_fc_qk/log.txt'))
# acc1, loss1, recon_loss1 =  (get_data('exp_hr/svd/deit_tiny_1e-4_5e-6_100_mix_head_fc_qk_hid_2/log.txt'))
# acc2, loss2, recon_loss2 =  (get_data('exp_hr/svd/deit_tiny_5e-4_5e-6_100_mix_head_fc_qk/log.txt'))


# acc0, loss0, recon_loss0 = extend_data(acc0, loss0, recon_loss0)
# acc1, loss1, recon_loss1 = extend_data(acc1, loss1, recon_loss1)
# acc2, loss2, recon_loss2 = extend_data(acc2, loss2, recon_loss2)

# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, figsize=(10, 5))

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']


# x = range(100)
# ax1.plot(x, acc0, label = "lr=1e-4 hid_dim=1")
# ax1.plot(x, acc1, label = "lr=1e-4 hid_dim=2")
# ax1.plot(x, acc2, label = "lr=5e-4 hid_dim=1")
# ax1.plot(x[:100], [72.2] * 100, '--', label = 'unpruned')
# ax1.title.set_text('Accuracy (Deit Tiny)')

# ax2.plot(x, loss0, label = "lr=1e-4 hid_dim=1")
# ax2.plot(x, loss1, label = "lr=1e-4 hid_dim=2")
# ax2.plot(x, loss2, label = "lr=5e-4 hid_dim=1")
# ax2.title.set_text('Test loss')

# ax3.plot(x, rec_loss0, label = "lr=1e-4 hid_dim=1")
# ax3.plot(x, rec_loss1, label = "lr=1e-4 hid_dim=2")
# ax3.plot(x, rec_loss2, label = "lr=5e-4 hid_dim=1")
# # ax3.plot([i+1 for i in range(len(rec_loss0))], rec_loss0, label='loss', c=colors[1])
# ax3.title.set_text('Reconstruction Loss')
# ax3.set_yscale('log')

# # plt.legend()
# leg = ax1.legend(fontsize=14, loc='lower right')
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(1)

# f.tight_layout()
# plt.savefig("./exp_hr/mix_head_fc_qk_deit_tiny.png")

######################### deit_tiny (low rank qk) end #########################

# deit_small (low rank qk + sparse)

acc0, loss0, recon_loss0 =  (get_data('exp_hr/lowrank_sparse/deit_tiny/lowrank_qk_hid_1_1e-5_100_info80/log.txt'))
acc1, loss1, recon_loss1 =  (get_data('exp_hr/lowrank_sparse/deit_tiny/lowrank_qk_hid_1_1e-5_100_info70/log.txt'))
acc2, loss2, recon_loss2 =  (get_data('exp_hr/lowrank_sparse/deit_tiny/lowrank_qk_hid_1_5e-4_100_info50/log.txt'))


acc0, loss0, recon_loss0 = extend_data(acc0, loss0, recon_loss0)
acc1, loss1, recon_loss1 = extend_data(acc1, loss1, recon_loss1)
acc2, loss2, recon_loss2 = extend_data(acc2, loss2, recon_loss2)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, figsize=(10, 5))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


x = range(100)
ax1.plot(x, acc0, label = "sparsity=0.8 hid_dim=1")
ax1.plot(x, acc1, label = "sparsity=0.7 hid_dim=1")
ax1.plot(x, acc2, label = "sparsity=0.5 hid_dim=1")
ax1.plot(x[:100], [72.2] * 100, '--', label = 'unpruned')
ax1.title.set_text('Accuracy (Deit Tiny)')

ax2.plot(x, loss0, label = "sparsity=0.8 hid_dim=1")
ax2.plot(x, loss1, label = "sparsity=0.7 hid_dim=1")
ax2.plot(x, loss2, label = "sparsity=0.5 hid_dim=1")
ax2.title.set_text('Test loss')

ax3.plot(x, recon_loss0, label = "sparsity=0.8 hid_dim=1")
ax3.plot(x, recon_loss1, label = "sparsity=0.7 hid_dim=1")
ax3.plot(x, recon_loss2, label = "sparsity=0.5 hid_dim=1")
# ax3.plot([i+1 for i in range(len(rec_loss0))], rec_loss0, label='loss', c=colors[1])
ax3.title.set_text('Reconstruction Loss')
ax3.set_yscale('log')

# plt.legend()
leg = ax1.legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1)

f.tight_layout()
plt.savefig("./exp_hr/lowrank_sparse_deit_tiny.png")
