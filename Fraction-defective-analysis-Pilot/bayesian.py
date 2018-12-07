# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 03:43:01 2018

@author: INFIA Protocol
"""

import numpy as np

k = 2 # slope

c = 5 # bias

s = 2 # noise standard deviation

import pandas as pd


x = pd.read_csv('testset.csv')
x = x.fillna(value=x.mean())
x = x.fillna(value=0)

#x = x.as_matrix()

#x1 = np.arange(10000)

#yy = k*x1 + c + s*np.random.randn(10000)

#firstline = True

#header = next(x)
#filewriter.writerow(header)

#x.columns = x.columns.strip()

x1 = x.drop(['dewptm','fog','hail','heatindexm','hum','precipm','pressurem','rain','snow','tempm','thunder','tornado','vism','wdird','wgustm','windchillm','wspdm'], axis=1)
#x1 = x1.dropna()

#x2 = x1.drop(1,0)
#x2 = x2.values

list_data = ['dewptm','fog','hail','heatindexm','hum','precipm','pressurem','rain','snow','tempm','thunder','tornado','vism','wdird','wgustm','windchillm','wspdm']

for i in list_data:
    buffer_data = i
    
    x2 = x[buffer_data]  
    #x2 = x3[buffer_data].drop(1,0)
    x2 = x2.values
    #x2 /= x2.max()
    
    firstline = True
    
    for xx in x2:
        if firstline:
            y = np.mean(xx)
            firstline = False
            continue
        y = np.vstack([y,(k*np.mean(xx) + c + s*np.random.randn(1))])
        
    y = y.reshape(y.shape[0],)
        
    X = x2.reshape(x2.shape[0],1)
        
    from bayespy.nodes import GaussianARD
    B = GaussianARD(0, 1e-6, shape=(X.shape[1],))
    from bayespy.nodes import SumMultiply
    F = SumMultiply('i,i', B, X)
        
    from bayespy.nodes import Gamma
    tau = Gamma(1e-3, 1e-3)
    Y = GaussianARD(F, tau)
    Y.observe(y)
    from bayespy.inference import VB
    Q = VB(Y, B, tau)
    #Q.update(repeat=100990)
    distribution = []
    result = []
    distribution= F.get_moments()
    for min_val,max_val in zip(distribution[0], distribution[1]):
        #mean = []
        mean = (min_val + max_val) / 2
        result.append(mean)
        #result = mean
        #x3 = []
        #x3 = pd.DataFrame({result:buffer_data})
        #x1 = x1.append(x3)
    x1[buffer_data] = result
                
print(x1)

dataframe = pd.DataFrame(x1)
dataframe.to_csv('distribution.csv', mode='a', header=True, index=False)

'''
CSV 챠트 그리기
'''
import matplotlib.pyplot as plt
import csv

f = pd.read_csv('distribution.csv')
df = f.drop(['datetime','conds','dire'], axis=1)
df = df.values

x = []
z = []
for row in df:
    x.append(row[0])
    z.append(row[1])

plt.plot(x, z, 'r')
plt.axis([-1, 30, -5, 1 ])
plt.show()


'''
import scipy.io

## Save the data

scipy.io.savemat('distribution.mat',{'x1': x1})
'''