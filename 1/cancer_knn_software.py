from kmodes.kmodes import KModes
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import csv
import pickle
import pandas as pd
from pandas import DataFrame
#import categorical_encoders as ce

#uci_dir_data='../../uci_datasets/classification_dataset/breast+cancer'
uci_dir_data='./breast+cancer'
img_data='breast_cancer.csv'

dataset_img=os.path.join(uci_dir_data,img_data)
dat=[]
with open(dataset_img,'r') as f:
	tmp=csv.reader(f)
	for line in tmp:
		dat.append(line)
raw_data=np.array(dat)

data=pd.DataFrame(raw_data)
#print(data)
data_encoded=pd.get_dummies(data=data,drop_first=False,dummy_na=False)
#print(data_encoded)
data_encoded.to_csv('./dummy_encoding.csv',sep=',')
target=data_encoded[data_encoded.columns[0:1]]
target=target.to_numpy()
#print(target)

data_point=data_encoded[data_encoded.columns[2:46]]
data_point=data_point.to_numpy()
#print(data_point)


#"""
x_train,x_test, y_train, y_test = train_test_split(data_point,target,test_size=0.3)

with open ('x_tr.pickle','wb')as f:
    pickle.dump(x_train,f)
with open ('x_te.pickle','wb')as f:
    pickle.dump(x_test,f)    
with open ('y_tr.pickle','wb')as f:
    pickle.dump(y_train,f) 
with open ('y_te.pickle','wb')as f:
    pickle.dump(y_test,f)
#"""

with open ('x_tr.pickle','rb')as f:
    x_train=pickle.load(f)
with open ('x_te.pickle','rb')as f:
    x_test=pickle.load(f)
with open ('y_tr.pickle','rb')as f:
    y_train=pickle.load(f)
with open ('y_te.pickle','rb')as f:
    y_test=pickle.load(f)

#neigh= KNeighborsClassifier(n_neighbors=5,metric='hamming')
#neigh.fit(x_train,y_train)
#score=neigh.score(x_test, y_test)
#print('score',score)
print('x_train',x_train.shape)
#print('x_train',x_train)


num_of_k=3
predict=[]
total_dist=[]
for dist_meas in x_test:
    dist_array_each_test=[]
    for waynum in range(0,200):
        HD=0
        for i,j in zip (dist_meas, x_train[waynum]):
            #if i == j:
                #HD=HD+1
            if (i==1) and (j==1):
                HD=HD+1
        dist_array_each_test.append(HD)
    total_dist.append(dist_array_each_test)
    
    nearest_vector=np.argsort(dist_array_each_test)[::-1]
    k_nearest=nearest_vector[0:num_of_k]
    label=[]
    for lab in k_nearest:
        label.append(y_train[lab])
    pred_label0=label.count(0)
    pred_label1=label.count(1)        
    tmp=[pred_label0,pred_label1]
    predict.append(np.argmax(tmp))


"""
hd=np.array(total_dist)
    
with open ('./ham_dist_cancer.csv','a') as f:
    wr=csv.writer(f)
    wr.writerow(hd[21])
    wr.writerow(predict)
    wr.writerow(y_train)
"""

count=0
y_test=np.squeeze(y_test)
print('test',y_test)
print('pred',predict)

for i,j in zip(predict,y_test):
    if i == j:
        count=count+1
acc=count/len(predict)
print('accuracy:',acc)


