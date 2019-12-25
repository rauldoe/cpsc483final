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
    # newU = np.delete(newU, changeIndex, axis = 1)

    newS = S.copy()
    changeIndex = range(featureCount, S.shape[1])
    newS[:, changeIndex] = 0
    # newS = np.delete(newS, changeIndex, axis = 0)
    # newS = np.delete(newS, changeIndex, axis = 1)

    newVh = Vh.copy()
    changeIndex = range(featureCount, Vh.shape[0])
    for row in range(featureCount, Vh.shape[0]):
        for col in range(len(newVh[row])):
            newVh[row][col] = 0
    # newVh = np.delete(newVh, changeIndex, axis = 0)

    return (newU.real, newS.real, newVh.real)

#X =np.array([[3, 1, 1], [-1, 3, 1]])
#X =np.array([[6, 2, 7, 9], [2, 3, 8, 10]])
#X =np.array([[6, 2, 7, 9, 11], [2, 3, 8, 10, 12], [63, 32, 73, 39, 113]])
U, S, Vh, recon = svd(X)
#print(recon)
U, S, Vh = reduceDimension(U, S, Vh, featureCount)
recon = U @ S @ Vh.T
recon = np.delete(recon, range(2, len(recon[0])), axis = 1)
#print(recon)

pc = np.delete(U, range(2, len(U[0])), axis = 1)
transformed = recon
# *** SVD END *** 

print("After SVD, Printing original data projected onto principal components")
processSummary(pc, original, transformed)

