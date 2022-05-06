import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import numpy as np
from progress.bar import Bar
from ase import Atoms
from itertools import combinations
from ase.db import connect
from ase.visualize import view
#from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
# Plot result
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')  
from itertools import cycle

from sklearn.cluster import Birch, DBSCAN, KMeans, MeanShift, estimate_bandwidth, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import random
#from yellowbrick.cluster import intercluster_distance
#from yellowbrick.cluster import silhouette_visualizer
#from yellowbrick.cluster.elbow import kelbow_visualizer

seed = 1234
random.seed(seed)
np.random.seed(seed)

import pickle
with open("/home/qzh/data_pre/data_dict.lst", 'rb') as fp:
    data_dict = pickle.load(fp)
#print(len(data_dict))

def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

figsize = (10, 8)
cols = 5
rows = 2

fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)

axs = trim_axs(axs, cols*rows)
ax_idx = 0


clst_dict = {}
total_clst_cnt = 0
#fig_data = []
for k in data_dict.keys():
    if len(k) != 2:
        continue
    patience = 0
    X = data_dict[k]
    if len(X) < 10000:
        continue
    # mean-shift
    label_unique = []
    bandwidth = estimate_bandwidth(X, quantile=0.015, n_jobs=-1,n_samples=10000, random_state=9)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    label_unique = np.unique(labels)
    #print(k, label_unique)
    #if(len(label_unique) > 1):
        #print('MS: ', k, len(label_unique), metrics.silhouette_score(X, labels,sample_size=100000), len(X))
    total_clst_cnt += len(label_unique)
    clst_dict[k] = ms.cluster_centers_

    #print(len(ms.cluster_centers_)) 
    #kelbow_visualizer(brc, X, k=(2,100), outpath="./data/birch/elbow.png")
    #silhouette_visualizer(brc(best_clst_cnts), X, colors='yellowbrick', outpath="./data/birch/sl.png")
    #intercluster_distance(brc(best_clst_cnts), X, outpath="./data/birch/inter.png")

    if len(k) == 2:
       ax = axs[ax_idx]
       ax_idx+=1
       fig_X = [[i] for i in range(4000)]
       fig_Y = [0 for i in range(4000)]
       for x in X:
           fig_Y[int(x[0]*1000)]+=1
       non_zero_X = [fig_X[i] for i in range(4000)]
       non_zero_Y = [fig_Y[i] for i in range(4000)]
       ax.set_title(k)
       ax.plot(list(np.array(non_zero_X)/1000.0), non_zero_Y)
       # print(results[0][4])
       for i in range(len(clst_dict[k])):
           #plt.axvline(x=clst_dict[0][i][0],ls="-",c="green")#添加垂直直线
           ax.plot([clst_dict[k][i][0], clst_dict[k][i][0]], [-max(non_zero_Y)/20, 0], linestyle=':')
       
print(total_clst_cnt)

#axs[5].set_ylabel('pair count')
#axs[7].set_xlabel('pair distance, ' + r'$\AA$')
plt.savefig('/home/qzh/data_pre/2atoms_down/ms_down.jpg')
plt.show()

#with open('/data/bak/qzh/e2e_reaction_test/julei_mini/shuju/slht_fig_data_02.txt', 'wb') as fp:
    #pickle.dump(fig_data, fp) 
#with open('/home/qzh/data/meanshift_0.015q/clst_dict.dct', 'wb') as fp:
#    pickle.dump(clst_dict, fp)
