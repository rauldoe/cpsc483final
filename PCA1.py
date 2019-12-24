
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
    return [X_pca, eigen_values, eigen_vectors]

def getFeatureVectors(eigenValues, eigenVectors, featureCount):

    featureVectors = []
    featureIndexes = np.argsort(eigenValues)
    primary = np.flip(featureIndexes[-1*featureCount:])

    for fIndex in primary:
        featureVectors.append(eigenVectors[fIndex])
    
    return featureVectors

# loading file
# datafile = 'PCAData.mat'
datafile = 'cars.mat'
points = scipy.io.loadmat( datafile )

# print shape of original data
X = points['X'] 
loaded = X.copy()
X = np.delete(X, range(7), axis=1)
print('Printing data in X...')
print(X)
# testing
# X = np.array([[1, 2, 5], [3, 4, 6]])
m, n = X.shape

print(m, n)
original = X.copy()

SumX = np.sum(X, axis =0)
meanX = np.mean(X,axis =0)
stdX = np.std(X,axis =0)

# print(meanX)
# print(stdX)
# print(SumX/50)

meanXReplicate = np.tile(meanX,(m,1))
stdXReplicate = np.tile(stdX,(m,1))
# print(meanXReplicate)

X = normalize(X,meanXReplicate,stdXReplicate)

# print(X)
# meanX = np.round(np.mean(X,axis =0),2)
# stdX = np.std(X,axis =0)
# print(meanX)
# print(stdX)

pcaInfo = pca(X)
pcaX = pcaInfo[0]
eigenValues = pcaInfo[1]
eigenVectors = pcaInfo[2]

fVectors = getFeatureVectors(eigenValues, eigenVectors, 2)
# print(pcaX)

m, n = pcaX.shape

print(m, n)

plt.scatter(pcaX[:,0], pcaX[:,1])
plt.plot((0, fVectors[0][0]), (0, fVectors[1][0]), c='red')
plt.plot((0, fVectors[0][1]), (0, fVectors[1][1]), c='green')
plt.show()

