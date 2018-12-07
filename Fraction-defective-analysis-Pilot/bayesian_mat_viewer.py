# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:49:49 2018

@author: INFIA Protocol
"""
import scipy
import numpy as np

mat = scipy.io.loadmat('distribution.mat')
## The observed data
##for i in mat:
#print(mat['distribution'][0])
#print(mat['distribution'][1])

print(mat.keys())
mat.shape()