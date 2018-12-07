# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 02:28:03 2018

@author: SAMSUNG
"""
#from sklearn.neural_network import MLPClassifier
import pandas as pd
#import numpy as np

x = pd.read_csv('testset.csv')
x = x.fillna(value=x.mean())
x = x.fillna(value=0)
