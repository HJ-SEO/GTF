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
"""
uci_dir_data='../uci_datasets/classification_dataset/breast+cancer'
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
"""

#use encoded excel file

"""

#data_encoded.to_csv('./onehot_encoding_zeropad.csv',sep=',')  #zero padding for the same feature size


#data_encoded=pd.read_csv('./onehot_encoding_nonzeropad.csv')#no zeropadding 


#data_encoded=pd.read_csv('./missing_col_onehot_encoding.csv')  #missing data feature deleted

npdata=data_encoded.to_numpy()
#####dont care 
doncare=[]
doncare.append(npdata[145])
doncare.append(npdata[163])
doncare.append(npdata[164])
doncare.append(npdata[183])
doncare.append(npdata[184])
doncare.append(npdata[233])
doncare.append(npdata[263])
doncare.append(npdata[264])
doncare.append(npdata[206])
#don_row=pd.DataFrame(doncare)

no_don=np.delete(npdata,264,axis=0)
no_don=np.delete(no_don,263,axis=0)
no_don=np.delete(no_don,233,axis=0)
no_don=np.delete(no_don,206,axis=0)
no_don=np.delete(no_don,184,axis=0)
no_don=np.delete(no_don,183,axis=0)
no_don=np.delete(no_don,164,axis=0)
no_don=np.delete(no_don,163,axis=0)
no_don=np.delete(no_don,145,axis=0)

no_don=pd.DataFrame(no_don)
#tmp=tmp.drop(tmp.columns[1])
#print(tmp)

#tmp.to_csv('./test.csv',sep=',')

target=no_don[no_don.columns[0:1]]
print(target)
target=target.to_numpy()

data_point=no_don[no_don.columns[1:43]]
#print(data_point)
data_point=data_point.to_numpy()

#x loc : 147,165,166,185,186,235,265,266,
#208

x_train,x_test, y_train, y_test = train_test_split(data_point,target,test_size=0.3)
x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)

#print(y_train)

don=pd.DataFrame(doncare)
don_target=don[don.columns[0:1]]
don_target=don_target.to_numpy()

don_data=don[don.columns[1:43]]
don_data=don_data.to_numpy()
idx=np.where(don_data=='1')
don_data[idx]=1

idx0=np.where(don_data=='0')
don_data[idx0]=0
#don_data=np.place(don_data,don_data=='0',0)
           
#print(don_data)



x_train=np.concatenate((x_train,don_data),axis=0)
y_train=np.concatenate((y_train,don_target),axis=0)
#print(y_train.shape)
#print(x_train)



with open ('x_tr.pickle','wb')as f:
    pickle.dump(x_train,f)
with open ('x_te.pickle','wb')as f:
    pickle.dump(x_test,f)    
with open ('y_tr.pickle','wb')as f:
    pickle.dump(y_train,f) 
with open ('y_te.pickle','wb')as f:
    pickle.dump(y_test,f)

"""

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
#print('x_train',x_train.shape)
#print('x_train',x_train)
#print('xtest',x_train)

num_of_k=3
predict=[]
total_dist=[]
for dist_meas in x_test:   

    dist_array_each_test=[]
    for waynum in range(0,202):
        HD=0
        for i,j in zip (dist_meas, x_train[waynum]):            
            if j=='x':
                HD=HD+1
            #elif (i == 1) and (j == 1):
                #HD=HD+1
            elif i == j:
                HD=HD+1
            #elif i!=j:
                #HD=HD+1
        dist_array_each_test.append(HD)
    #total_dist.append(dist_array_each_test)
    #print(total_dist)
    nearest_vector=np.argsort(dist_array_each_test)[::-1]
    #nearest_vector=np.argsort(total_dist)[::-1]
    #nearest_vector=np.argsort(dist_array_each_test)
    k_nearest=nearest_vector[0:num_of_k]
    label=[]
    for lab in k_nearest:
        label.append(y_train[lab])
    pred_label0=label.count(0)
    pred_label1=label.count(1)        
    tmp=[pred_label0,pred_label1]
    predict.append(np.argmax(tmp))




count=0
y_test=np.squeeze(y_test)
print('test',y_test)
print('pred',predict)

for i,j in zip(predict,y_test):
    if i == j:
        count=count+1
acc=count/len(predict)
print('accuracy:',acc)

