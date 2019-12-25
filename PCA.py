import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load *.mat files

dataFile = 'cars.mat'
featureCount = 2

def normalize(X,meanX,stdX):
    
    
    X = (X-meanX)/stdX
        
    return X

def loadData(dataFile):
    # loading file
    # datafile = 'PCAData.mat'
    # datafile = 'cars.mat'
    points = scipy.io.loadmat( dataFile )

    # print shape of original data
    X = points['X'] 
    # loaded = X.copy()
    X = np.delete(X, range(7), axis=1)
    #print('Printing data in X...')
    #print(X)
    m, n = X.shape

    #print(m, n)
    original = X.copy()

    SumX = np.sum(X, axis = 0)
    meanX = np.mean(X, axis = 0)
    stdX = np.std(X, axis = 0)

    # print(meanX)
    # print(stdX)
    # print(SumX/50)

    meanXReplicate = np.tile(meanX,(m,1))
    stdXReplicate = np.tile(stdX,(m,1))
    # print(meanXReplicate)

    X = normalize(X,meanXReplicate,stdXReplicate)

    return (X, original)

X, original = loadData(dataFile)

# *** PCA START *** 
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

def getFeatureMatrix(eigenValues, eigenVectors, featureCount):

    # Take the number of features with the max values in front
    featureMatrix = []
    featureIndexes = np.argsort(eigenValues)
    primary = np.flip(featureIndexes[-1*featureCount:])

    for fIndex in primary:
        featureMatrix.append(eigenVectors[fIndex])
    
    return np.array(featureMatrix)

# print(X)
# meanX = np.round(np.mean(X,axis =0),2)
# stdX = np.std(X,axis =0)
# print(meanX)
# print(stdX)

pcaInfo = pca(X)
pcaX = pcaInfo[0]
eigenValues = pcaInfo[1]
eigenVectors = pcaInfo[2]

featureMatrix = getFeatureMatrix(eigenValues, eigenVectors, featureCount)
featureMatrixT = featureMatrix.T

# We want the transformed matrix to be a row_count x 2
transformedMatrix = (np.dot(featureMatrix, original.T)).T
# *** PCA END *** 

print(transformedMatrix)

m, n = transformedMatrix.shape

print(m, n)

# plt.scatter(pcaX[:,0], pcaX[:,1])
# plt.plot((0, fVectors[0][0]), (0, fVectors[1][0]), c='red')
# plt.plot((0, fVectors[0][1]), (0, fVectors[1][1]), c='green')
# plt.show()

