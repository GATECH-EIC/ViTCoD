import pickle
import matplotlib.pyplot as plt
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
    try:
        data = open(filename, 'rt')
        lines = data.read().split('\n')
        for l in lines:
            if l != '':
                dictionary = parse(l)
                acc.append(dictionary['test_acc1'])
                loss.append(dictionary['test_loss'])
        data.close()
    except:
        print("Error")
    return [float(x) for x in acc], [float(x) for x in loss]

acc1, loss1 =  (get_data('/home/zs19/deit/info9_5e-6/log.txt'))
# print (acc1)
acc2, loss2 =  (get_data('/home/zs19/deit/info9_1e-5/log.txt'))
acc3, loss3 =  (get_data('/home/zs19/deit/info9_2e-5/log.txt'))
acc3.extend([None, None])
loss3.extend([None, None])
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
x = range(40)
ax1.plot(x, acc1, label = "lr=5e-6")
ax1.plot(x, acc2, label = "lr=1e-5")
ax1.plot(x, acc3, label = "lr=2e-5")
ax1.title.set_text('Accuracy')
ax2.plot(x, loss1, label = "lr=5e-6")
ax2.plot(x, loss2, label = "lr=1e-5")
ax2.plot(x, loss3, label = "lr=2e-5")
ax2.title.set_text('Test loss')
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')
# ax1.plot(rng, line3, label = "sparsity=90.36 lr=1e-5")
plt.legend()
plt.savefig("ratio_based")