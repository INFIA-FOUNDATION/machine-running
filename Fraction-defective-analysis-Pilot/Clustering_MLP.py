# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 02:21:23 2018

@author: INFIA Protocol
"""

'''
Create Clustering 
'''
import sys

import pandas as pd

from sklearn.cluster import KMeans

import numpy as np

import matplotlib.pyplot as plt

import scipy

'''
x = []

mean = []

mat = scipy.io.loadmat('distribution.mat')

for min_val,max_val in zip(mat['distribution'][0],mat['distribution'][1]):

    mean = (min_val + max_val) / 2

    x.append([ min_val, max_val, mean])

'''
x = pd.read_csv('distribution.csv')
#x = x.dropna()
x1= x.drop(['datetime','conds','dire'], axis=1)
#x1 = x1.dropna()
x1 = x1.values

#datetime = x['datetime']  

#conds = x['conds']  

#dire = x['dire']

# 데이터 프레임을 구성(표로 만든다)

# column명들은 아래와 같다

#x1 = x['dewptm','fog','hail','heatindexm','hum','precipm','pressurem','rain','snow','tempm','thunder','tornado','vism','wdird','wgustm','windchillm','wspdm']  

#x1 = x1.values

cluster_num = 4

# KMeans의 파라미터 --> n_cluster는 군집의 개수를 의미(여기선 3개), 나머지 파라미터는 고정값

km = KMeans(n_clusters=cluster_num,

    init='k-means++',

    n_init=100,

    max_iter=400,

    tol=1e-04,

    random_state=0)

y_km = km.fit_predict(x1)

# 군집 결과를 데이터 프레임에 추가한다

x['CLUSTERS'] = y_km

'''
x = np.asarray(x)
plt.scatter(x[y_km==0,1],

            x[y_km==0,2],

            s=25,

            c='lightgreen',

            marker='s',

            label='Cluster 1')

plt.scatter(x[y_km==1,1],

            x[y_km==1,2],

            s=25,

            c='orange',

            marker='o',

            label='Cluster 2')

plt.scatter(x[y_km==2,1],

            x[y_km==2,2],

            s=25,

            label='Cluster 3')

plt.scatter(km.cluster_centers_[:,0],

            km.cluster_centers_[:,1],

            s=200,

            marker='*',

            c='red',

            label='centroids')

plt.legend()

plt.grid()

plt.show()


cluster_distribution = []

cluster_distribution = df.values

import scipy.io

## Save the data

scipy.io.savemat('cluster_distribution.mat',{'cluster_distribution': cluster_distribution})

'''

#dataframe = pd.DataFrame(x)

x.to_csv('cluster_distribution.csv', header=True, index=False)

"""
Cluster 결과 파일 분할

train_list, test_list, train_data, test_data = [],[],[],[]

with open('cluster_distribution.csv') as f:

    line_counter =0

    while 1:

        data = f.readline()

        if not data:

            break 

        if line_counter == 0:

            header = data.split(",") # 맨 첫 줄은 header로 저장

        else:

            if line_counter <= 8000:

                train_list.append(data.split(","))

            else:

                test_list.append(data.split(","))

        line_counter += 1

    train_data.append(header)

    train_data.append(train_list)

    dataframe = pd.DataFrame(train_data)

    dataframe.to_csv('cluster_train.csv', mode='a', header=True, index=False)

    test_data.append(header)

    test_data.append(train_list)

    dataframe = pd.DataFrame(test_data)

    dataframe.to_csv('cluster_test.csv', mode='a', header=True, index=False)
"""

cluster_distribution = pd.read_csv('cluster_distribution.csv')

cluster_train = cluster_distribution[0:80000]

cluster_test =  cluster_distribution[80000:]

cluster_train.to_csv('cluster_train.csv', header=True, index=False)

cluster_test.to_csv('cluster_test.csv', header=True, index=False)

"""

Create Clustering_MLP 

"""

from sklearn.neural_network import MLPClassifier

'''
mat = scipy.io.loadmat('cluster_distribution.mat')

for i in mat['cluster_distribution']:

    train_x.append([ i[0], i[1], i[2] ])

    train_y.append(i[3])

    test_x.append([ i[0], i[1], i[2] ])

    test_y.append(i[3])

'''

train_orig = pd.read_csv('cluster_train.csv')

train = train_orig.drop(['datetime','conds','dire'], axis=1)

train1 = train.drop(['CLUSTERS'], axis=1)

#train1 = train1.dropna()

train2 = train['CLUSTERS']

train_x = train1.values

train_y = train2.values

test_orig = pd.read_csv('cluster_test.csv')

test = test_orig.drop(['datetime','conds','dire'], axis=1)

test1 = test.drop(['CLUSTERS'], axis=1)

#test1 = test1.dropna()

test2 = test['CLUSTERS']

test_x = test1.values

test_y = test2.values

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(24,12), random_state=1)

clf.fit(train_x,train_y)

predicted_y = clf.predict(test_x)

accuracy = len([predicted_y[n] for n in range(len(predicted_y)) if predicted_y[n] == test_y[n]]) / len(test_y) * 100

#print('Accuracy is {}'.format(accuracy))

test_orig['Predict_Cluster'] = predicted_y

#dataframe = pd.DataFrame(test)

test_orig.to_csv('clusterMLP_distribution.csv', header=True, index=False)

'''
CSV 챠트 그리기
'''
import matplotlib.pyplot as plt
import csv

f = pd.read_csv('clusterMLP_distribution.csv')
df = f.drop(['datetime','conds','dire'], axis=1)
df = df.values

x = []
z = []
for row in df:
    x.append(row[0])
    z.append(row[17])

plt.plot(x, z, 'r')
plt.axis([-1, 30, -5, 1 ])
plt.show()


'''

clusterMLP_distribution = []

for i, j in zip( mat['cluster_distribution'], predicted_y ):

    clusterMLP_distribution.append([ i[0], i[1], i[2], i[3], j ])

import scipy.io

## Save the data

scipy.io.savemat('clusterMLP_distribution.mat',{'clusterMLP_distribution': clusterMLP_distribution})

'''
		
