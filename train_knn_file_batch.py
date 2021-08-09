import pickle
import h5py
import numpy as np
import random
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

seed = 1234
random.seed(seed)
np.random.seed(seed)

file_cnt = 99
BoC_size = 0

try:
    
    clf = joblib.load('./shuju/model/clf_400_knn.pkl')     
except:
    print('初始化模型')
    #clf = svm.LinearSVC()
    #clf = svm.SVC(kernel='linear')
    #clf = svm.libsvm(kernel='linear')
    #clf = tree.DecisionTreeClassifier()
    clf = KNeighborsClassifier(weights='distance',n_neighbors = 50)

batch_size = 1024
max_test_acc = 0

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

    
for i in range(1, file_cnt+1):
    dataset = None
    data_arr = None
    label_arr = None
    f = h5py.File('/data/qzh/julei/dataset_new_' + str(i) + '.hdf5', 'r')
    print('start to load data.')
    dataset = f['dset1'][:]
    print(len(dataset),len(dataset[0]))
    print(dataset[0][:10])

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
    print('input BoC length is: ', len(dataset[0])-1)
    BoC_size = len(dataset[0])-1
    #pca = KernelPCA(n_components=8, kernel='linear', n_jobs = 2)
    #pca = PCA(n_components=400)
    #data = pca.fit_transform(data)
    #data = normalization(data)
    #print(data[0])
    data_arr = np.array(data, dtype=np.float32)
    label_arr = np.array(label)     

    dataset = None
    data = None
    label = None
    print(i, data_arr.shape, label_arr.shape)
    f.close()
    

    print('Data load finished.')
 
    train_sets = data_arr[:int(len(data_arr)*0.8)]
    vali_sets = data_arr[int(len(data_arr)*0.8): int(len(data_arr)*0.9)]
    test_sets = data_arr[int(len(data_arr)*0.9):]
    train_labels = label_arr[:int(len(data_arr)*0.8)]
    vali_labels = label_arr[int(len(data_arr)*0.8): int(len(data_arr)*0.9)]
    test_labels = label_arr[int(len(data_arr)*0.9):]
    
    print(test_labels)
    
	for epoch in range(1):  # loop over the dataset multiple times
        # train
        for i in range(0, len(train_labels), batch_size):
            # get the inputs; data is a list of [inputs, labels]
            # inputs = torch.from_numpy(train_sets[i:min(i+batch_size, len(label))]).to('cpu')
            # labels = torch.from_numpy(train_labels[i:min(i+batch_size, len(label))]).to('cpu')
            inputs = train_sets[i:i+batch_size]
            labels = train_labels[i:i+batch_size]

            try:
                clf.fit(inputs, labels)
            except:
                print(i, '都为1')
            outputs = clf.predict(inputs) 
            
            print("训练集：", accuracy_score(labels,outputs))

	# valid
        with torch.no_grad():
            N = len(vali_sets)
            for i in range(0, N, batch_size):
                #inputs = torch.from_numpy(vali_sets[i:i+batch_size]).to('cpu')
                #labels = torch.from_numpy(vali_labels[i:i+batch_size]).to('cpu')
                inputs = vali_sets[i:i+batch_size]
                labels = vali_labels[i:i+batch_size]
                vali_outputs = clf.predict(inputs)
        print("验证集：", accuracy_score(labels,vali_outputs))
        
	# test
        with torch.no_grad():
            N = len(test_sets)
            for i in range(0, N, batch_size):
                #inputs = torch.from_numpy(test_sets[i:i+batch_size]).to('cpu')
                #labels = torch.from_numpy(test_labels[i:i+batch_size]).to('cpu')
                inputs = test_sets[i:i+batch_size]
                labels = test_labels[i:i+batch_size]
                test_outputs = clf.predict(inputs)

        print("测试集：", accuracy_score(labels,test_outputs))
        if accuracy_score(labels,test_outputs) > max_test_acc:
            max_test_acc = accuracy_score(labels,test_outputs)
            joblib.dump(clf, './shuju/model/clf_400_knn.pkl') 


print('Finished Training')
