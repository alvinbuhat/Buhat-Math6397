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


A = imread(os.path.join('dog.jpg')) #load the image of the dog
X = np.mean(A, -1); 


U, S, VT = np.linalg.svd(X,full_matrices=True)
S = np.diag(S)

r = 500 #any number less than 1500 (Chosen rank < m)

I = np.linalg.inv(U[:r,:r]) @ U[:r,:r]
J = U[:r,:r] @ np.linalg.inv(U[:r,:r])
K = np.linalg.norm(I-np.eye(r))



print('the SVD is U=', U)
print('S=', S)
print('VT=', VT)
print('We choose r=',r)
#Showing U*U = I
print('So we have the identity matrix from U*U=',I)
#Showing U*U != UU*
print('However, UU* doesnt generate an identity matrix as shown:',J)

print('with error', K)

#error plot
C = [np.linalg.norm(U[:r,:r] @ np.linalg.inv(U[:r,:r]) - np.eye(r)) for r in range(U.shape[0])]

plt.yscale('log')
plt.plot(C);





    
