import h5py
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import argparse 
seed = 1234
random.seed(seed)
np.random.seed(seed)

def load_data():
    global cur_file_id, BoC_size, data_arr, label_arr

    f = h5py.File('./data/dataset_new_' + str(cur_file_id) + '.hdf5', 'r')
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
   
    if cur_file_id == 1:
        print('input BoC length is: ', len(dataset[0])-1)
        BoC_size = len(dataset[0])-1
        data_arr = np.array(data, dtype=np.float32)
        label_arr = np.array(label)
    else:
        data_arr = np.concatenate((data_arr, np.array(data, dtype=np.float32)), axis=0)
        label_arr = np.concatenate((label_arr, np.array(label)), axis=0)
    
    print(cur_file_id, data_arr.shape, label_arr.shape)
    cur_file_id += 1
    f.close()

def create_model(s):
    print('初始化'+s+'模型')
    if s == 'SDG':
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    elif s == 'KNN':
        clf = KNeighborsClassifier(weights='distance')
    elif s == 'Tree':
        clf = tree.DecisionTreeClassifier()
    elif s == 'LinearSVC':
        clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
    elif s == 'SVC':
        clf = svm.SVC(kernel='linear')
    return clf

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--model', type=str, default='SDG',
            help='choose the trained model.please enter: SDG, KNN, Tree, LinearSVC, or SVC')

    
    return parser.parse_args()

def main():
    
    for i in range(5):
        load_data()
    print('Data load finished.')

    train_sets = data_arr[:int(len(data_arr)*0.9)]
    test_sets = data_arr[int(len(data_arr)*0.9):]
    train_labels = label_arr[:int(len(data_arr)*0.9)]
    test_labels = label_arr[int(len(data_arr)*0.9):]

    args = get_arguments()
    s=args.model
    clf = create_model(s)
    clf.fit(train_sets, train_labels)
    outputs = clf.predict(train_sets) 
    print("训练集：", accuracy_score(train_labels,outputs))

    test_outputs = clf.predict(test_sets) 
    print("测试集：", accuracy_score(test_labels,test_outputs))

if __name__ == "__main__":
    cur_file_id = 1
    BoC_size = 0
    dataset = None
    data_arr = None
    label_arr = None
    main()
