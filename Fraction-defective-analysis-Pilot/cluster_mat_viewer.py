# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 03:43:25 2018

@author: INFIA Protocol
"""
import scipy

mat = scipy.io.loadmat('cluster_distribution.mat')
## The observed data
##for i in mat:
#print(mat['cluster_distribution'][0])
#print(mat['cluster_distribution'][1])
#print(mat['cluster_distribution'][2])

for i in mat['cluster_distribution']:
    print(i[0], i[1], i[2], i[3])