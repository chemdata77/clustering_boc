import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
import pandas as pd 
import random
seed = 1234
random.seed(seed)
np.random.seed(seed)

import pickle
with open("./data/data_dict_mini.lst", 'rb') as fp:
    data_dict = pickle.load(fp)
print(len(data_dict))

clst_dict = {}
total_clst_cnt = 0

for k in data_dict.keys():
    patience = 0
    X = data_dict[k]
    if len(X) < 10000:
        continue
    data = pd.DataFrame(X[:])
    print(data)
    from pycaret.clustering import *
    exp_clu101 = setup(data, normalize = True, session_id = 123)
    kmeans = create_model('kmeans')
    kmean_results = assign_model(kmeans)

    plot_model(kmeans, plot = 'elbow',save = True)

    plot_model(kmeans, plot = 'silhouette',save = True)


    label_unique = []
    labels = kmeans.labels_
    label_unique = np.unique(labels)
    total_clst_cnt += len(label_unique)
    clst_dict[k] = kmeans.cluster_centers_
    print(k,'break')
    break

       
print(total_clst_cnt)

with open('./data/pycaret/clst_dict.dct', 'wb') as fp:
    pickle.dump(clst_dict, fp)
