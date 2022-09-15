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
from sklearn import decomposition


plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 10})

#A
#Finding the SVD of VORTALL

mat_contents = scipy.io.loadmat(os.path.join('CYLINDER_ALL.mat')) 

vortall = mat_contents['VORTALL']
Uall = mat_contents['UALL']
Vall = mat_contents['VALL']

m, n = *mat_contents['m'], *mat_contents['n']

U, S, VT = np.linalg.svd(vortall, full_matrices=False) 

Z = vortall
pca = decomposition.PCA(n_components = 1)
pca.fit(Z)
Z = pca.transform(Z)

#singular value
plt.semilogy(S)

#spectrum
plt.figure(figsize=(10,5))
plt.imshow(U[:,0].reshape(*n,*m))
plt.figure(figsize=(10,5))
plt.imshow(U[:,1].reshape(*n,*m))
plt.figure(figsize=(10,5))
plt.imshow(U[:,2].reshape(*n,*m))
plt.figure(figsize=(10,5))
plt.imshow(U[:,3].reshape(*n,*m))
plt.figure(figsize=(10,5))
plt.imshow(U[:,4].reshape(*n,*m));

#B
#Write a code to plot for various truncated values
for r in (2,3,10): #predetermined values but can do iteration if necessary


    X = U[:,:r] @ np.diag(S)[:r,:r] @ VT[:r,:r] #Truncation

    pca = decomposition.PCA(n_components = 1) #PCA to compute for fluid flow movie
    pca.fit(X)
    X = pca.transform(X)
    y = pca.explained_variance_ratio_
    err = (np.linalg.norm(Z - X))**2

    print("When r = " , r , " there are " , y*100 , " % captured flow energy with Squared Frobenius norm", err )
    plt.figure(figsize=(10,5))
    plt.imshow(X[:,0].reshape(*n,*m))


#C
#Fix r and compute for truncated SVD

#truncated matrix
r = 10

Xtrunc = U[:,:r] @ np.diag(S)[:r,:r] @ VT[:r,:r]
W = np.diag(S)[:r,:r] @ VT[:r,:r]

#method of snapshot
V_S2_VT = vortall[:,:r].T @ vortall[:,:r]
S2_MoS, V_MoS = np.linalg.eig(V_S2_VT)
S_MoS = np.sqrt(abs(S2_MoS))
U_MoS = vortall[:,:r] @ V_MoS @ np.linalg.inv(np.diag(S_MoS))

#plot comparison
#k=1
plt.figure(figsize=(10,5))
plt.set_cmap('twilight')
plt.imshow((U_MoS @ np.diag(S_MoS) @ V_MoS)[:,1].reshape(*n,*m))


plt.figure(figsize=(10,5))
plt.set_cmap('twilight')
plt.imshow((U[:,:r]@W[:r,:r])[:,1].reshape(*n,*m))

#k=10
plt.figure(figsize=(10,5))
plt.set_cmap('twilight')
plt.imshow((U_MoS @ np.diag(S_MoS) @ V_MoS)[:,10].reshape(*n,*m))


plt.figure(figsize=(10,5))
plt.set_cmap('twilight')
plt.imshow((U[:,:r]@W[:r,:r])[:,10].reshape(*n,*m))
    



#Build linear regression model




#Advance the state w_k
