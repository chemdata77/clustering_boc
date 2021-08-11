import h5py
import os
import numpy as np
file_cnt = 100
idx = 0
for i in range(1,file_cnt+1,5):
    idx+=1
    for j in range(i,i+5):
        f = h5py.File('/data/qzh/julei/dataset_new_'+str(j)+'.hdf5','r')
        dataset = f['dset1'][:]
        if j%5 == 1:
            ret = dataset
        else:
            ret = np.vstack((ret,dataset))
    os.system('rm -rf ' + '/data/qzh/data_v2/dataset_new_' + str(idx) + '.hdf5') 
    f = h5py.File('dataset_new_' + str(idx) + '.hdf5', 'w')
    f.create_group('/grp1') # or f.create_group('grp1')
    f.create_dataset('dset1', compression='gzip', data=np.array(ret)) # or f.create_dataset('/dset1', data=data)
    f.close()
   
 
        
        
