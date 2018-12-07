# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:15:00 2018

@author: INFIA Protocol
"""

#import random

#import sys

#import scipy

import numpy as np

import pandas as pd

from sklearn.neural_network import MLPClassifier

train = pd.read_csv('cluster_train.csv')

#train = train.dropna()

train1= train.drop(['datetime','conds','dire'], axis=1)

#train1 = train1.dropna()

train1 = train1.values

test = pd.read_csv('cluster_test.csv')

buffer = pd.read_csv('cluster_test.csv')

#test = test.dropna()

test1= test.drop(['datetime','conds','dire'], axis=1)

#test1 = test1.dropna()

test1 = test1.values

'''
mat = scipy.io.loadmat('clusterMLP_distribution.mat')

# 학습 데이터를 받아와서 저장

for i in mat['clusterMLP_distribution']:

    train_data.append([i[0], i[1], i[2], i[3], i[4]])

for i in mat['clusterMLP_distribution']:

    test_data.append([i[0], i[1], i[2], i[3], i[4]])

# 쿼리로 받은 데이터를 고객 ID별로 묶어준다

whole_data =[]

    

for line in train_data:

    whole_data.append([line[0], line[1], line[2]])

'''

'''
defective_x = []

not_defective_x=[]

defective_label = 0

not_defective_label = 1

    

for line in whole_data:

    for n in range(len(line) ,2,-1):

        if n == len(line):

            defective_x.append(line[n-10000:n] )

        else:

            not_defective_x.append(line[n-10000:n])




random.shuffle(not_defective_x)

not_defective_x = not_defective_x[0:len(defective_x)]

X = []

X.extend(defective_x)

X.extend(not_defective_x)

Y=[]

Y.extend([defective_label] * len(defective_x))

Y.extend([not_defective_label] * len(not_defective_x))

'''

accuracy_list = []

defective_001_label = 1

defective_002_label = 2

defective_003_label = 3

defective_004_label = 4

defective_005_label = 5

defective_006_label = 6

defective_007_label = 7

defective_008_label = 8

defective_009_label = 9

defective_010_label = 10

defective_011_label = 11

defective_012_label = 12

defective_013_label = 13

defective_014_label = 14

defective_015_label = 15

defective_016_label = 16

defective_017_label = 17

defective_018_label = 18

defective_019_label = 19

defective_020_label = 20

defective_021_label = 21

defective_022_label = 22

defective_023_label = 23

defective_024_label = 24

defective_025_label = 25

defective_026_label = 26

defective_027_label = 27

defective_028_label = 28

defective_029_label = 29

defective_030_label = 30

defective_031_label = 31

defective_032_label = 32

defective_033_label = 33

defective_034_label = 34

defective_035_label = 35

defective_036_label = 36

defective_037_label = 37

defective_038_label = 38

defective_039_label = 39

defective_040_label = 40

#X = []

#X.append(train2)

X = train1

#Y = []

train_yy = train['conds']

#train_y = train_y.dropna()

train_yy = train_yy.values

test_yy = test['conds']

#train_y = train_y.dropna()

test_yy = test_yy.values

#train_yy = np.array([])

#train_yy = np.zeros((1,80000))

#train_yy = train_y

#np.sort(buffer_yy, axis=None, kind='quicksort')

print('print start-->')

train_yy[train_yy == 'Blowing Sand'] = 1

train_yy[train_yy == 'Clear'] = 2

train_yy[train_yy == 'Drizzle'] = 3

train_yy[train_yy == 'Fog'] = 4

train_yy[train_yy == 'Funnel Cloud'] = 5

train_yy[train_yy == 'Haze'] = 6

train_yy[train_yy == 'Heavy Fog'] = 7

train_yy[train_yy == 'Heavy Rain'] = 8

train_yy[train_yy == 'Heavy Thunderstorms and Rain'] = 9

train_yy[train_yy == 'Heavy Thunderstorms with Hail'] = 10

train_yy[train_yy == 'Light Drizzle'] = 11

train_yy[train_yy == 'Light Fog'] = 12

train_yy[train_yy == 'Light Freezing Rain'] = 13

train_yy[train_yy == 'Light Hail Showers'] = 14

train_yy[train_yy == 'Light Haze'] = 15

train_yy[train_yy == 'Light Rain'] = 16

train_yy[train_yy == 'Light Rain Showers'] = 17

train_yy[train_yy == 'Light Sandstorm'] = 18

train_yy[train_yy == 'Light Thunderstorm'] = 19

train_yy[train_yy == 'Light Thunderstorms and Rain'] = 20

train_yy[train_yy == 'Mist'] = 21

train_yy[train_yy == 'Mostly Cloudy'] = 22

train_yy[train_yy == 'Overcast'] = 23

train_yy[train_yy == 'Partial Fog'] = 24

train_yy[train_yy == 'Partly Cloudy'] = 25

train_yy[train_yy == 'Patches of Fog'] = 26

train_yy[train_yy == 'Rain'] = 27

train_yy[train_yy == 'Rain Showers'] = 28

train_yy[train_yy == 'Sandstorm'] = 29

train_yy[train_yy == 'Scattered Clouds'] = 30

train_yy[train_yy == 'Shallow Fog'] = 31

train_yy[train_yy == 'Smoke'] = 32

train_yy[train_yy == 'Squalls'] = 33

train_yy[train_yy == 'Thunderstorm'] = 34

train_yy[train_yy == 'Thunderstorms and Rain'] = 35

train_yy[train_yy == 'Thunderstorms with Hail'] = 36

train_yy[train_yy == 'Unknown'] = 37

train_yy[train_yy == 'Volcanic Ash'] = 38

train_yy[train_yy == 'Widespread Dust'] = 39

train_yy[train_yy == 0] = 40

print('print end-->')

print('print start-->')

test_yy[test_yy == 'Blowing Sand'] = 1

test_yy[test_yy == 'Clear'] = 2

test_yy[test_yy == 'Drizzle'] = 3

test_yy[test_yy == 'Fog'] = 4

test_yy[test_yy == 'Funnel Cloud'] = 5

test_yy[test_yy == 'Haze'] = 6

test_yy[test_yy == 'Heavy Fog'] = 7

test_yy[test_yy == 'Heavy Rain'] = 8

test_yy[test_yy == 'Heavy Thunderstorms and Rain'] = 9

test_yy[test_yy == 'Heavy Thunderstorms with Hail'] = 10

test_yy[test_yy == 'Light Drizzle'] = 11

test_yy[test_yy == 'Light Fog'] = 12

test_yy[test_yy == 'Light Freezing Rain'] = 13

test_yy[test_yy == 'Light Hail Showers'] = 14

test_yy[test_yy == 'Light Haze'] = 15

test_yy[test_yy == 'Light Rain'] = 16

test_yy[test_yy == 'Light Rain Showers'] = 17

test_yy[test_yy == 'Light Sandstorm'] = 18

test_yy[test_yy == 'Light Thunderstorm'] = 19

test_yy[test_yy == 'Light Thunderstorms and Rain'] = 20

test_yy[test_yy == 'Mist'] = 21

test_yy[test_yy == 'Mostly Cloudy'] = 22

test_yy[test_yy == 'Overcast'] = 23

test_yy[test_yy == 'Partial Fog'] = 24

test_yy[test_yy == 'Partly Cloudy'] = 25

test_yy[test_yy == 'Patches of Fog'] = 26

test_yy[test_yy == 'Rain'] = 27

test_yy[test_yy == 'Rain Showers'] = 28

test_yy[test_yy == 'Sandstorm'] = 29

test_yy[test_yy == 'Scattered Clouds'] = 30

test_yy[test_yy == 'Shallow Fog'] = 31

test_yy[test_yy == 'Smoke'] = 32

test_yy[test_yy == 'Squalls'] = 33

test_yy[test_yy == 'Thunderstorm'] = 34

test_yy[test_yy == 'Thunderstorms and Rain'] = 35

test_yy[test_yy == 'Thunderstorms with Hail'] = 36

test_yy[test_yy == 'Unknown'] = 37

test_yy[test_yy == 'Volcanic Ash'] = 38

test_yy[test_yy == 'Widespread Dust'] = 39

test_yy[test_yy == 0] = 40

print('print end-->')

#Y = np.array([])

#Y = np.zeros((1,80000))

#Y = np.vstack([Y, train_yy])       

Y = train_yy

Y = Y.astype('int')

test_yy = test_yy.astype('int')

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500,200,100,24,12), random_state=0)

clf.fit(X,Y)

print('done clf !!!!!!!!!!!!!')
    
'''

id_data = []

tmp_id=[]

    

for line in test_data:

    whole_data.append([line[0], line[1], line[2]])

    

test_x = []

tmp = []




for line in whole_data:

    for n in range(len(line) ,2,-1):

        if n == len(line):

            tmp.append(line[n-10000:n])

            break

'''

test_x = test1

#predicted_y = clf.predict(test_x)

predicted_yy = clf.predict(test_x)

accuracy = len( [p for p, y in zip(predicted_yy, test_yy) if p == y] ) / len(predicted_yy) *100

accuracy_list.append( '{}''s MLP result accuracy is {:.2f}%'.format(test_x, accuracy) )

print('Accuracy is {}'.format(accuracy))

#predicted_yy = np.array([]) 

#predicted_yy = np.zeros((1,80000))

#predicted_yy = pd.DataFrame(predicted_y)

predicted_yy = predicted_yy.astype('object')

predicted_yy[predicted_yy == 1] = 'Blowing Sand'

predicted_yy[predicted_yy == 2] = 'Clear'

predicted_yy[predicted_yy == 3] = 'Drizzle'

predicted_yy[predicted_yy == 4] = 'Fog'

predicted_yy[predicted_yy == 5] = 'Funnel Cloud'

predicted_yy[predicted_yy == 6] = 'Haze'

predicted_yy[predicted_yy == 7] = 'Heavy Fog'

predicted_yy[predicted_yy == 8] = 'Heavy Rain'

predicted_yy[predicted_yy == 9] = 'Heavy Thunderstorms and Rain'

predicted_yy[predicted_yy == 10] = 'Heavy Thunderstorms with Hail'

predicted_yy[predicted_yy == 11] = 'Light Drizzle'

predicted_yy[predicted_yy == 12] = 'Light Fog'

predicted_yy[predicted_yy == 13] = 'Light Freezing Rain'

predicted_yy[predicted_yy == 14] = 'Light Hail Showers'

predicted_yy[predicted_yy == 15] = 'Light Haze'

predicted_yy[predicted_yy == 16] = 'Light Rain'

predicted_yy[predicted_yy == 17] = 'Light Rain Showers'

predicted_yy[predicted_yy == 18] = 'Light Sandstorm'

predicted_yy[predicted_yy == 19] = 'Light Thunderstorm'

predicted_yy[predicted_yy == 20] = 'Light Thunderstorms and Rain'

predicted_yy[predicted_yy == 21] = 'Mist'

predicted_yy[predicted_yy == 22] = 'Mostly Cloudy'

predicted_yy[predicted_yy == 23] = 'Overcast'

predicted_yy[predicted_yy == 24] = 'Partial Fog'

predicted_yy[predicted_yy == 25] = 'Partly Cloudy'

predicted_yy[predicted_yy == 26] = 'Patches of Fog'

predicted_yy[predicted_yy == 27] = 'Rain'

predicted_yy[predicted_yy == 28] = 'Rain Showers'

predicted_yy[predicted_yy == 29] = 'Sandstorm'

predicted_yy[predicted_yy == 30] = 'Scattered Clouds'

predicted_yy[predicted_yy == 31] = 'Shallow Fog'

predicted_yy[predicted_yy == 32] = 'Smoke'

predicted_yy[predicted_yy == 33] = 'Squalls'

predicted_yy[predicted_yy == 34] = 'Thunderstorm'

predicted_yy[predicted_yy == 35] = 'Thunderstorms and Rain'

predicted_yy[predicted_yy == 36] = 'Thunderstorms with Hail'

predicted_yy[predicted_yy == 37] = 'Unknown'

predicted_yy[predicted_yy == 38] = 'Volcanic Ash'

predicted_yy[predicted_yy == 39] = 'Widespread Dust'

predicted_yy[predicted_yy == 40] = '0'

#predicted_yyy = np.array([])

#predicted_yyy = np.zeros((1,80000)) 

#predicted_yyy = np.vstack([predicted_yyy, predicted_yy])        

buffer['Predict_Defective'] = predicted_yy

buffer.to_csv('MLP_distribution.csv', header=True, index=False)