#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:46:38 2022

@author: alvinbuhat
"""

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8] #matlab operator set size of image


A = imread(os.path.join('dog.jpg')) #load the image
X = np.mean(A, -1); 

U, S, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

x_0 = np.linalg.norm(A)

RE = [abs(((np.linalg.norm(U[:,:r] @ S[0:r,:r] @ VT[:r,:]))/x_0) - 1) for r in range(1,len(S))]




plt.title('Relative Reconstruction error, Missed Variance and Cumulative Sum')
plt.xlabel('r')
plt.plot(1 - np.array(RE), color='blue', label='Relative Reconstruction Error')
plt.plot(1 - (np.array(RE)**2), color='red', label='Variance')
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)), color='black', label='Cumulative Sum')
plt.legend();

plt.tight_layout()

print('RE recaptures the 99% of variance when r is around 20, 99% of cumulative sum when r is at most 20, although note that they have different starting points')
