#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:34:57 2022

@author: alvinbuhat
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 18})

mat_contents = scipy.io.loadmat(os.path.join('allFaces.mat')) #Load the Yale B image
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

allPersons = np.zeros((n*6,m*6))
count = 0


for j in range(6):
    for k in range(6):
        allPersons[j*n : (j+1)*n, k*m : (k+1)*m] = np.reshape(faces[:,np.sum(nfaces[:count])],(m,n)).T
        count += 1
        
img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
plt.show()


U, S, VT = np.linalg.svd(allPersons, full_matrices=False) #compute the economy SVD

#Using method of snapshots

VS2VT = faces[:,:36].T @ faces[:,:36]  #XTX

#computing SVD using MOS

S2_MOS, V_MOS = np.linalg.eig(VS2VT) #sigma squared and V from MOS
U_MOS = faces[:,:36] @ V_MOS @ np.linalg.inv(np.diag(np.sqrt(abs(S2_MOS)))) #U from MOS = X@V@Sigma from MOS inverse

#Singular Values Comparison

plt.figure(figsize=(10,5))
plt.plot(S, label='SVD singular values')
plt.plot(np.sqrt(abs(S2_MOS)),  label='MoS singular values')
plt.yscale('log')
plt.legend();


#Compare the first 10 left singular vectors using each method and Compare further vectors
#I am not sure how to do this part
