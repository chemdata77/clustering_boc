import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import numpy as np
from progress.bar import Bar
from ase import Atoms
from itertools import combinations
from ase.db import connect
from ase.visualize import view
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
# Plot result
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
from itertools import cycle

from sklearn.cluster import Birch, DBSCAN, KMeans, MeanShift, estimate_bandwidth, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np
import random

seed = 1234
random.seed(seed)
np.random.seed(seed)

import pickle
with open("/data/qzh/julei/data_dict_mini.lst", 'rb') as fp:
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
# import numpy as np
# from sklearn.datasets.samples_generator import make_blobs
# # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
# X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state = 9)
# X, y = make_blobs(n_samples=1000, n_features=3, centers=[[-1,-1,-1], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state = 9)
# X, y = make_blobs(n_samples=1000, n_features=1, centers=[[-1], [0], [1], [2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state = 9)

fig_data = []
for k in data_dict.keys():
    patience = 0
    X = data_dict[k]
    if len(X) < 10000:
        continue
    # mean-shift
    label_unique = []
    #bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=10000, random_state=9)
    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #ms.fit(X)
    #labels = ms.labels_
    #label_unique = np.unique(labels)
    #if(len(label_unique) > 1):
        #print('MS: ', k, len(label_unique), metrics.silhouette_score(X, labels, sample_size=10000), len(X))
    #total_clst_cnt += len(label_unique)
    #clst_dict[k] = ms.cluster_centers_

    #mini-batch k means
    best_score = 0
    best_clst_cnts = 2
    res = []
    for n_clst in range(2, 100):
        if n_clst > 5 and n_clst % 2 == 0:
            continue 
        km = MiniBatchKMeans(n_clusters=n_clst, random_state=9)
        y_pred = km.fit_predict(X)

        #brc = Birch(n_clusters = n_clst, threshold = 0.3, branching_factor = 50)
        #y_pred = brc.fit_predict(X)

        # print(metrics.calinski_harabaz_score(X, y_pred))
        slht = metrics.silhouette_score(X, y_pred, sample_size=1000)
        res.append([k, slht])
        # slht = metrics.calinski_harabasz_score(X, y_pred)
        if(slht > best_score):
            best_score = slht
            best_clst_cnts = n_clst
            patience = 0
        else:
            patience += 1
        if patience > 5 and n_clst>10:
            break
        #print('res',res)
    #print('KM: ', k, best_clst_cnts, best_score, len(X))

    #print('brc: ', k, best_clst_cnts, best_score, len(X))
    km = MiniBatchKMeans(n_clusters=best_clst_cnts, random_state=9)
    y_pred = km.fit_predict(X)
    slht = metrics.silhouette_score(X, y_pred, sample_size=1000)
    #print('slht',slht)
    #brc = Birch(n_clusters = best_clst_cnts, threshold = 0.3, branching_factor = 50)
    #y_pred = brc.fit_predict(X)
    fig_data.append(res)
    total_clst_cnt += max(best_clst_cnts, len(label_unique))
    if best_clst_cnts > len(label_unique):
        clst_dict[k] = km.cluster_centers_
        #clst_dict[k] = brc.cluster_centers_

    if len(k) == 2:
       ax = axs[ax_idx]
       ax_idx+=1
       fig_X = [[i] for i in range(10000)]
       fig_Y = [0 for i in range(10000)]
       for x in X:
           fig_Y[int(x[0]*1000)]+=1
       non_zero_X = [fig_X[i] for i in range(10000) if fig_Y[i] >= int(max(fig_Y)/20)]
       non_zero_Y = [fig_Y[i] for i in range(10000) if fig_Y[i] >= int(max(fig_Y)/20)]
       ax.set_title(k[0] + '-' + k[1])
       ax.plot(list(np.array(non_zero_X)/1000.0), non_zero_Y)
       # print(results[0][4])
       for i in range(len(clst_dict[k])):
           #plt.axvline(x=clst_dict[0][i][0],ls="-",c="green")#添加垂直直线
           ax.plot([clst_dict[k][i][0]/1000, clst_dict[k][i][0]/1000], [0, max(non_zero_Y)], linestyle=':')
       
       

    #DBSCAN
    #label_unique = []
    #epsilon(X,10000)
    #db = DBSCAN(eps=epsilon(X, 500), min_samples=500)
    #db.fit(X)
    #print(db.components_)
    #labels = db.labels_
    #label_unique = np.unique(labels)
    #print(len(labels), label_unique)
    #if(len(label_unique) > 1):
        #print('DB: ', k, len(label_unique), metrics.silhouette_score(X, labels), len(X))
    #total_clst_cnt += len(label_unique)
    #clst_dict_01[k] = labels
    #print(labels[:20])
    
    # k-mean
    #best_score = 0
    #best_clst_cnts = 2
    #for n_clst in range(2, 100):
        #if n_clst > 5 and n_clst % 2 == 0:
            #continue 
        #km = KMeans(n_clusters=n_clst, random_state=9)
        #y_pred = km.fit_predict(X)
        # print(metrics.calinski_harabaz_score(X, y_pred))
        #slht = metrics.silhouette_score(X, y_pred, sample_size=1000)
        # slht = metrics.calinski_harabasz_score(X, y_pred)
        #if(slht > best_score):
            #best_score = slht
            #best_clst_cnts = n_clst
            #patience = 0
        #else:
            #patience += 1
        #if patience > 5 and n_clst>10:
            #break
    #print('KM: ', k, best_clst_cnts, best_score, len(X))
    #km = KMeans(n_clusters=best_clst_cnts, random_state=9)
    #y_pred = km.fit_predict(X)
    #total_clst_cnt += max(best_clst_cnts, len(label_unique))
    #if best_clst_cnts > len(label_unique):
        #clst_dict[k] = km.cluster_centers_

print(total_clst_cnt)

axs[5].set_ylabel('pair count')
axs[7].set_xlabel('pair distance, ' + r'$\AA$')
#plt.savefig('./shuju/slht.jpg')
plt.show()

with open('/data/bak/qzh/e2e_reaction_test/julei_mini/shuju/slht_fig_data_02.txt', 'wb') as fp:
    pickle.dump(fig_data, fp) 
with open('/data/bak/qzh/e2e_reaction_test/julei_mini/shuju/clst_dict_test_87.dct', 'wb') as fp:
    pickle.dump(clst_dict, fp)
