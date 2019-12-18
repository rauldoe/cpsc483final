
# clear; python PCA1.py;
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:07:40 2019

@author: shlakhanpal
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load *.mat files

def normalize(X,meanX,stdX):
    
    
    X = (X-meanXReplicate)/stdXReplicate
        
    return X
    
def pca(X): 

    m, n = X.shape 
    
    # Compute covariance matrix 
    C = np.dot(X.T, X) / (m-1)
    
    # Eigen decomposition 
    eigen_values, eigen_vectors = np.linalg.eig(C) 
     
    print (eigen_vectors)
    
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vectors)
    return X_pca

datafile = 'PCAData.mat'
points = scipy.io.loadmat( datafile )

X = points['X'] 
print('Printing data in X...')
print(X)
m, n = X.shape

print(m, n)

SumX = np.sum(X, axis =0)
meanX = np.mean(X,axis =0)
stdX = np.std(X,axis =0)

print(meanX)
print(stdX)
#print(SumX/50)

meanXReplicate = np.tile(meanX,(m,1))
stdXReplicate = np.tile(stdX,(m,1))
#print(meanXReplicate)
X = normalize(X,meanXReplicate,stdXReplicate)

print(X)
meanX = np.round(np.mean(X,axis =0),2)
stdX = np.std(X,axis =0)

print(meanX)
print(stdX)

pcaX = pca(X)
print(pcaX)

m, n = pcaX.shape

print(m, n)

