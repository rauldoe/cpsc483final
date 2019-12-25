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
    m, _ = X.shape

    #print(m, n)
    original = X.copy()

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

def processSummary(pc, original, transformed):
    print("Display the principal components from both PCA and SVD.")
    print(pc)
    print("Project the original data instances (points) onto the principal components resulting from both SVD and PCA. ")
    print("Plot the two principal components from SVD ( PC1, labeled as such, capturing the maximum variance from the data, and PC2, labeled as such, the next one) and the projections onto them of the original data.")
    plt.scatter(original[:,0], original[:,1])
    plt.plot(pc[:,0], pc[:,1], c='red')
    plt.scatter(transformed[:,0], transformed[:,1], c='green')
    # plt.plot((0, fVectors[0][1]), (0, fVectors[1][1]), c='green')
    plt.show()

X, original = loadData(dataFile)
print("Printing original data")
print(X)

# *** PCA START *** 
def pca(X): 

    m, _ = X.shape 
    
    # Compute covariance matrix 
    C = np.dot(X.T, X) / (m-1)
    
    # Eigen decomposition 
    eigen_values, eigen_vectors = np.linalg.eig(C) 
     
    # print (eigen_vectors)
    
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

pcaX, eigenValues, eigenVectors = pca(X)

featureMatrix = getFeatureMatrix(eigenValues, eigenVectors, featureCount)
featureMatrixT = featureMatrix.T

# We want the transformed matrix to be a row_count x 2
transformedMatrix = (np.dot(featureMatrix, original.T)).T

#print(transformedMatrix)

m, n = transformedMatrix.shape

#print(m, n)

# plt.scatter(pcaX[:,0], pcaX[:,1])
# plt.plot((0, fVectors[0][0]), (0, fVectors[1][0]), c='red')
# plt.plot((0, fVectors[0][1]), (0, fVectors[1][1]), c='green')
# plt.show()

pc = featureMatrix
transformed = transformedMatrix
# *** PCA END *** 

print("After PCA, Printing original data projected onto principal components")
processSummary(pc, original, transformed)