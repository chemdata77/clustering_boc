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
import numpy as np
import random
from yellowbrick.cluster import intercluster_distance
from yellowbrick.cluster import silhouette_visualizer
from yellowbrick.cluster.elbow import kelbow_visualizer

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

for k in data_dict.keys():
    X = data_dict[k]
    if len(X) < 10000:
        continue
    if len(X)>100000:
        X = random.sample(X,100000)
    #DBSCAN
    #epsilon(X,10000)
    db = DBSCAN(eps=0.05, min_samples=5)
    db.fit(X)
    #print(db.components_)
    labels = db.labels_
    #-1是噪声数据
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    print(k,n_clust) 
    ans = []
    for i in range(n_clust):
        res= []
        for j in range(len(labels)):
            if labels[j] == i:
                res.append(j)
                #print(res)
        ans.append(res)
    center = []
    for p in ans:
        X_km = []
        for q in p:
            X_km.append(X[q])
        #print(X_km)

        km = KMeans(n_clusters=1, random_state=9)
        y_pred = km.fit_predict(X_km)
        center.append(km.cluster_centers_[0])

    total_clst_cnt += n_clust
    clst_dict[k] = np.array(center)

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
            #plt.axvline(x=clst_dict[0][i][0],ls="-",c="green")
            ax.plot([clst_dict[k][i][0], clst_dict[k][i][0]], [-max(non_zero_Y)/20, 0], linestyle=':')

print(total_clst_cnt)

#axs[5].set_ylabel('pair count')
#axs[7].set_xlabel('pair distance, ' + r'$\AA$')
plt.savefig('/home/qzh/data_pre/dbscan_0.05.5/DBSCAN.jpg')
plt.show()

with open('/home/qzh/data_pre/dbscan_0.05.5/clst_dict_db.dct', 'wb') as fp:
    pickle.dump(clst_dict, fp)
