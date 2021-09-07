
import multiprocessing
import math
import numpy as np
from progress.bar import Bar
from ase import Atoms
from itertools import combinations
from ase.db import connect
from ase.visualize import view
from sklearn.cluster import AffinityPropagation

import numpy as np
import random

seed = 1234
random.seed(seed)
np.random.seed(seed)

atom_names = ['H', 'C', 'O', 'N']
atom_dict = {'H': 0, 'C':1, 'O':2, 'N':3}
data_dict = {}

# find all keys of data_dict
debug = False
clst_atom_cnt_min = 2
clst_atom_cnt_max = 4
max_clst_diameter = 5 # A

clst_sym_lst = [] # list all sym of clst

for cnts in range(clst_atom_cnt_max-1):
    clst_sym_lst_tmp = []
    if len(clst_sym_lst) == 0:
        for nm1 in atom_names:
            for nm2 in atom_names:
                clst_sym_lst_tmp.append(''.join(sorted(nm1+nm2)))
    else:
        for nm1 in clst_sym_lst:
            for nm2 in atom_names:
                clst_sym_lst_tmp.append(''.join(sorted(nm1+nm2)))
    clst_sym_lst += clst_sym_lst_tmp
    clst_sym_lst = list(set(clst_sym_lst))

#print('clst_sym_lst',len(clst_sym_lst),clst_sym_lst)

for clst in clst_sym_lst:
    data_dict[clst] = []
 #print(data_dict)

db = connect('./data_3.5/qm9.db')
# rows = list(db.select('F<1', sort='id')) # 131722 no F molecules
rows = list(db.select('F=0', sort='id')) # 131722 no F molecules
random.shuffle(rows)
# rows = rows[:20000]
# rows = list(db.select('id<200'))
#import pickle
#with open('smalldata_5000.lst', 'rb') as fp:
    #rows = pickle.load(fp)

#print(len(rows))
#print(rows[0].id)
#print(rows[:3])

# find all value (tuple of all distances) of data_dict

def multi_thd_reac(row):
    atoms = row.toatoms()
    #print('row atoms',atoms)
    data_dict_tmp = []
    #if(row.id % 1000 == 1):
        #print(row.id)
    for atom_cnts in range(clst_atom_cnt_min, clst_atom_cnt_max+1):
        for idx_tup in list(combinations(range(len(atoms)), atom_cnts)):
            #print('idx_tup: ', idx_tup)
            clst_vaild = True
            clst_hash = []
            for idx_pair in list(combinations(list(idx_tup), 2)):
                #print('idx_pair: ', idx_pair)
                #print('idx_pair[0]',idx_pair[0])
                #print('idx_pair[1]',idx_pair[1])
                d = atoms.get_distance(idx_pair[0], idx_pair[1])
                if(d > max_clst_diameter):
                    clst_vaild =False
                    break
                # d = 1-1/(d**2) * 1000
                # d = math.log(d-0.95) * 1000
                clst_hash.append((''.join(sorted(atoms[idx_pair[0]].symbol+atoms[idx_pair[1]].symbol)), d))
                #print('clst_hash',clst_hash)
            if clst_vaild:
                clst_hash = sorted(clst_hash)
                #print('clst_hash: ', clst_hash)
                dd_key = ''.join(sorted(list(atoms[idx].symbol for idx in idx_tup)))
                #print('dd_key', dd_key)
                dd_value = [tup[1] for tup in clst_hash]
                #print('dd_value', dd_value)
                data_dict_tmp.append((dd_key, dd_value))
                #print('data_dict_tmp',data_dict_tmp)
    return data_dict_tmp

# reac_lst = []
# for row in rows:
#     reac_lst = multi_thd_reac(row)
#     print(multi_thd_reac(row))
#     break
pool = multiprocessing.Pool(20)
reac_lst = pool.map(multi_thd_reac, rows)
#print('reac_lst',reac_lst[:10])
for clst_lst in reac_lst:
    #print('clst_lst',clst_lst)
    for clst in clst_lst:
        #print(clst)
        if len(data_dict[clst[0]]) < 1e6:
            data_dict[clst[0]].append(clst[1])

# print(data_dict)
import pickle
with open('./data_5/data_dict.lst', 'wb') as fp:
    pickle.dump(data_dict, fp)
