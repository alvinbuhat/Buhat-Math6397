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
# Convert RGB to grayscale

U, S, VT = np.linalg.svd(X,full_matrices=True)
S = np.diag(S)

r = 10 #any number less than 1500

I = np.linalg.inv(U[:r,:r]) @ U[:r,:r]
J = U[:r,:r] @ np.linalg.inv(U[:r,:r])
K = abs(I-J)



print('the SVD is U=', U)
print('S=', S)
print('VT=', VT)
print('We choose r=',r)
print('So we have the identity matrix from U*U=',I)
print('However, UU* doesnt generate an identity matrix as shown:',J)

print('with error', K)

#plt.figure(1)

for k in range(1,r):
    A = np.linalg.inv(U[:k,:k]) @ U[:k,:k]
    B = U[:k,:k] @ np.linalg.inv(U[:k,:k])
    C = abs(I-J)
    
plt.plot(C) 





    