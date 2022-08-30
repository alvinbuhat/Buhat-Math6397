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
X = np.mean(A, -1); #mean of array from the image #mean around -1

#r = 5

U, S, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

#Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]

x_0 = np.linalg.norm(A)

#x = np.linalg.norm(U @ S @ VT)

#RE = x_0/x - 1

#RE**2

k=100
#A = [((x_0/np.linalg.norm(U[:,:r] @ S[0:r,:r] @ VT[:r,:])) - 1) for r in range(1,k)] #RE
#B = [((x_0/np.linalg.norm(U[:,:r] @ S[0:r,:r] @ VT[:r,:])) - 1)**2 for r in range(1,k)] #RE^2
#plt.figure(1) #relative error and relative error ^ 2
#plt.plot(A)
#plt.plot(B)
#plt.figure(2) #cumulative sum
#plt.plot(np.cumsum(np.diag(S))) 
#plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))

frac = [(x_0/(np.linalg.norm(U[:,:r] @ S[0:r,:r] @ VT[:r,:]))-1) for r in range(len(S))]



plt.figure(figsize=(15,5))


plt.title('Relative Reconstruction error, Missed Variance and Cumulative Sum')
plt.xlabel('r')
plt.plot(1-np.array(frac), color='blue', label='Relative Reconstruction Error')
plt.plot(1-np.array(frac)**2, color='red', label='Variance')
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)), color='black', label='Cumulative Sum')
plt.legend();

plt.tight_layout()