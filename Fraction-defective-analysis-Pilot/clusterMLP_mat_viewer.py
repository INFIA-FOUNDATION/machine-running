# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 04:37:34 2018

@author: INFIA Protocol
"""

import scipy

mat = scipy.io.loadmat('clusterMLP_distribution.mat')
## The observed data
##for i in mat:
#print(mat['cluster_distribution'][0])
#print(mat['cluster_distribution'][1])
#print(mat['cluster_distribution'][2])

for i in mat['clusterMLP_distribution']:
    print(i)