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

#random100x100 matrix
A = np.random.normal(size=(100,100))

U, S, VT = np.linalg.svd(A, full_matrices=True)

#compute the SVD of the matrix and plot the singular values
plt.boxplot(np.array([np.linalg.svd(np.random.normal(size=(100,100)), full_matrices=True)[1] for i in range(100)]))

#function to get the median of sigmas
def smedian(S):
        res = []
        for i in range( S.shape[0]):
            res.append(np.median(S[:i+1]))
        return res
    

#plot of mean and median for different matrix size
for i in (50, 200, 500, 1000):
    #addl defns
    S1 = np.linalg.svd(np.random.normal(size=(i,i)), full_matrices=True)[1]
    u = len(S1)
    rng = np.arange(1, u + 1)

    plt.figure(i)
    plt.title(str(i)+' by ' + str(i))
    plt.xlabel('r')
    mean = np.cumsum(S1)/rng #command to get the mean 
    plt.plot(mean, label='mean')
    median =  smedian(S1) 
    plt.plot(rng, median, label='median')
    plt.legend()
 
