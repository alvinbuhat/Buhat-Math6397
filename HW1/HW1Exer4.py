#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:01:13 2022

@author: alvinbuhat
"""
#from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
#import os
plt.rcParams['figure.figsize'] = [16, 8] #matlab operator set size of image


A = np.random.normal(size=(100,100))

U, S, VT = np.linalg.svd(A, full_matrices=True)


plt.boxplot(np.array([np.linalg.svd(np.random.normal(size=(100,100)), full_matrices=True)[1] for i in range(100)]))

#def my_median(S):
 ##   res = []
   ##    res.append(np.median(S[:i+1]))
    #return res


for i in (50, 200, 500, 1000):
    S = np.linalg.svd(np.random.normal(size=(i,i)), full_matrices=True)[1]
    plt.figure(i)
    plt.title(str(i)+' by' + str(i))
    plt.xlabel('r')
    mean = np.cumsum(S)/np.arange(1,len(S)+1)
   # plt.plot(mean, label='mean')
    median = np.median(S[0:i])
    plt.plot(median, label='median')
    plt.legend()
