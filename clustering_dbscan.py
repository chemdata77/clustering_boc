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
with open("./data_3.5/data_dict.lst", 'rb') as fp:
    data_dict = pickle.load(fp)
#print(len(data_dict))
for k in data_dict.keys():
    X = data_dict[k]
    print(k,len(X))
print('over')
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
    #DBSCAN
    #epsilon(X,10000)
    db = DBSCAN(eps=500, min_samples=500)
    db.fit(X)
    print(db.components_)
    labels = db.labels_
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    
    if(len(label_unique) > 1):
        print('DB: ', k, n_clust, metrics.silhouette_score(X, labels), len(X))
    ans = []
    for i in range(n_clust):
        for j in range(len(labels)):
            if labels[j] == i:
                res= []
                res.append(j)
        ans.append(res)
    center = []
    for p in ans:
        for q in p:
            X_km = X[p]

        km = KMeans(n_clusters=1, random_state=9)
        y_pred = km.fit_predict(X_km)
        center += km.cluster_centers_

    total_clst_cnt += n_clust
    clst_dict[k] = center

    if len(k) == 2:
        ax = axs[ax_idx]
        ax_idx+=1
        fig_X = [[i] for i in range(4000)]
        fig_Y = [0 for i in range(4000)]
        for x in X:
            fig_Y[int(x[0]*1000)]+=1
        non_zero_X = [fig_X[i] for i in range(4000)]
        non_zero_Y = [fig_Y[i] for i in range(4000)]
        ax.set_title(k[0] + '-' + k[1])
        ax.plot(list(np.array(non_zero_X)/1000.0), non_zero_Y)
        # print(results[0][4])
        for i in range(len(clst_dict[k])):
            #plt.axvline(x=clst_dict[0][i][0],ls="-",c="green")
            ax.plot([clst_dict[k][i][0], clst_dict[k][i][0]], [0, max(non_zero_Y)], linestyle=':')

print(total_clst_cnt)

axs[5].set_ylabel('pair count')
axs[7].set_xlabel('pair distance, ' + r'$\AA$')
plt.savefig('./data/birch/birch.jpg')
plt.show()

with open('./data_3.5/dbscan/clst_dict_db.dct', 'wb') as fp:
    pickle.dump(clst_dict, fp)
