import h5py
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

seed = 1234
random.seed(seed)
np.random.seed(seed)

file_cnt = 5
step_cnt = 1e6

BoC_size = 0

dataset = None
data_arr = None
label_arr = None
for i in range(1, file_cnt+1):
    f = h5py.File('./data/dataset_new_' + str(i) + '.hdf5', 'r')
    print('start to load data.')
    dataset = f['dset1'][:]
    print(len(dataset),len(dataset[0]))

    print('start to shuffle data.')
    np.random.shuffle(dataset) # the data distribution is not uniform last 10% maybe all 1, so must shuffle here
    zero_cnt = 0
    data = []
    label = []
    for j in range(len(dataset)):
        if(dataset[j][-1]<0):
            # change -1 to 0, using softmax
            dataset[j][-1] = 0
            zero_cnt += 1
        data.append(dataset[j][0:-1])
        label.append(int(dataset[j][-1]))
    print(zero_cnt, 'start to cat data')
   
    if i == 1:
        print('input BoC length is: ', len(dataset[0])-1)
        BoC_size = len(dataset[0])-1
        data_arr = np.zeros((int(file_cnt * step_cnt), BoC_size))
        label_arr = np.zeros(int(file_cnt * step_cnt), dtype=np.int)
    else:
        pass
    data_arr[int((i-1)*step_cnt):int(i*step_cnt)] = np.array(data, dtype=np.float32)
    label_arr[int((i-1)*step_cnt):int(i*step_cnt)] = np.array(label)     

    dataset = None
    data = None
    label = None
    print(i, data_arr.shape, label_arr.shape)
    f.close()

train_sets = data_arr[:int(len(data_arr)*0.9)]
test_sets = data_arr[int(len(data_arr)*0.9):]
train_labels = label_arr[:int(len(data_arr)*0.9)]
test_labels = label_arr[int(len(data_arr)*0.9):]
del data_arr
del label_arr
print(test_labels)

clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(train_sets, train_labels)
outputs = clf.predict(train_sets) 
print("训练集：", accuracy_score(train_labels,outputs))

test_outputs = clf.predict(test_sets) 
print("测试集：", accuracy_score(test_labels,test_outputs))
