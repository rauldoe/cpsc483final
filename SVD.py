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

# *** SVD START *** 
def svd(X):
    aat = X @ X.T
    eigenValues_aat, U = np.linalg.eig(aat)
    S = np.diag(np.sqrt(eigenValues_aat))
    Vh = X.T @ U @ np.linalg.inv(S)
    recon = None
    #recon = U @ S @ Vh.T
    
    return (U, S, Vh, recon)

def reduceDimension(U, S, Vh, featureCount):
    
    newU = U.copy()
    changeIndex = range(featureCount, U.shape[1])
    newU[:, changeIndex] = 0

    newS = S.copy()
    changeIndex = range(featureCount, S.shape[1])
    newS[:, changeIndex] = 0

    newVh = Vh.copy()
    for row in range(featureCount, Vh.shape[0]):
        for col in range(len(newVh[row])):
            newVh[row][col] = 0
    
    return (newU, newS, newVh)

#X =np.array([[3, 1, 1], [-1, 3, 1]])
#X =np.array([[6, 2, 7, 9], [2, 3, 8, 10]])
#X =np.array([[6, 2, 7, 9, 11], [2, 3, 8, 10, 12], [63, 32, 73, 39, 113]])
U, S, Vh, recon = svd(X)
#print(recon)
U, S, Vh = reduceDimension(U, S, Vh, featureCount)
recon = U @ S @ Vh.T
print(recon)

