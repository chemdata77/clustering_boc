from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
np.random.seed(4711)

import pickle
with open("/home/qzh/data_pre/data_dict.lst", 'rb') as fp:
    data_dict = pickle.load(fp)

for k in data_dict.keys():
    X = np.array(data_dict[k])
    print(X.shape)
    Z = linkage(X, 'ward')

    plt.figure(figsize=(10, 8))
    #plt.title('Hierarchical Clustering Dendrogram')
    #plt.xlabel('sample index')
    #plt.ylabel('distance')
    #dendrogram(Z,leaf_rotation=90., leaf_font_size=8., )
    dendrogram(
                Z,
                    truncate_mode='lastp',  # show only the last p merged clusters
                        p=20,  # show only the last p merged clusters
                        leaf_font_size=12.,
                            leaf_rotation=90.,
                                    show_contracted=True,  # to get a distribution impression in truncated branches
                                    )
    plt.savefig('/home/qzh/data_pre/'+k+'_truncated_v6.jpg')
    plt.show()

    break
