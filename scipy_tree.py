from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
np.random.seed(4711)

import pickle
with open("/data2/qzh/data_3.5/data_dict.lst", 'rb') as fp:
    data_dict = pickle.load(fp)

for k in data_dict.keys():
    X = np.array(data_dict[k])
    print(X.shape)
    Z = linkage(X, 'ward')

    plt.figure(figsize=(250, 100))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z,leaf_rotation=90., leaf_font_size=8., )
    plt.show()
    break
