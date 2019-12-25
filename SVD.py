import numpy as np

def getLambdas(eigenValues, shape):
    lambdaMatrix = np.zeros(shape)
    for idx, eigenValue in enumerate(eigenValues):
        lambdaMatrix[idx][idx] = eigenValue

    return np.sqrt(lambdaMatrix)

def sortEigen(eigenValues, eigenVectors):

    sortedIndexes = np.argsort(eigenValues)[::-1]
    return (eigenValues[sortedIndexes], eigenVectors[:,sortedIndexes])

decimalPlaces = 5

A = np.array([[5, 1], [3, 3]])

eValues, eVectors = np.linalg.eig(A)


eVectors = eVectors.T
#print(eVectors)
# print (eValues)

# print(np.dot(A, eVectors[0].T))
# print(np.diag(eValues))
lambdas = np.diag(eValues)
V = eVectors.T

V_inv = np.linalg.inv(V)

# assert A == V.dot(lambdas).dot(V_inv)
print(V.dot(lambdas).dot(V_inv))




# A = np.array([[6, 2, 7, 9], [2, 3, 8, 10]])
A = np.array([[3, -2], [1, 4]])
A = np.array([[3, 2, 2], [2, 3, -2]])
# eigVals, eigVecs = np.linalg.eig(A)

# A * (A Transpose)
aat = np.dot(A, A.T)
#print(aat)
eigVals_aat, eigVecs_aat = np.linalg.eig(aat)
eigVals_aat = np.sqrt(np.round(eigVals_aat, decimals=decimalPlaces))
eigVecs_aat = eigVecs_aat.T
eigVals_aat, eigVecs_aat = sortEigen(eigVals_aat, eigVecs_aat)

# (A Transpose) * A
ata = np.dot(A.T, A)
#print(ata)
eigVals_ata, eigVecs_ata = np.linalg.eig(ata)
eigVals_ata = np.sqrt(np.round(eigVals_ata, decimals=decimalPlaces))
eigVecs_ata = eigVecs_ata.T
eigVals_ata, eigVecs_ata = sortEigen(eigVals_ata, eigVecs_ata)


lambdaMatrix = getLambdas(eigVals_aat, A.shape)
#print(lambdaMatrix)

V = eigVecs_aat
UT = eigVecs_ata

# assert A == V.dot(lambdas).dot(V_inv)
print(V.dot(lambdaMatrix).dot(UT.T))

u, s, vh = np.linalg.svd(A, full_matrices=False)

#print(u @ np.diag(s) @ vh)

# A = np.array([[3, 1, 1], [-1, 3, 1]])
# A = np.array([[6, 2, 7, 9], [2, 3, 8, 10]])
# temp = A.dot(A.T)
# S, U = np.linalg.eig(temp)
# S = np.diag(np.sqrt(S))
# V = A.T.dot(U).dot(np.linalg.inv(S))
# recon = np.dot(U, S).dot(V.T)
# print(recon)

A = np.array([[3, 1, 1], [-1, 3, 1]])
A = np.array([[6, 2, 7, 9], [2, 3, 8, 10]])
A = np.array([[6, 2, 7, 9, 11], [2, 3, 8, 10, 12], [63, 32, 73, 39, 113]])

aat = np.dot(A, A.T)
eigVals_aat, eigVecs_aat = np.linalg.eig(aat)
lambdaMatrix = np.diag(np.sqrt(eigVals_aat))
V = A.T.dot(eigVecs_aat).dot(np.linalg.inv(lambdaMatrix))
recon = np.dot(eigVecs_aat, lambdaMatrix).dot(V.T)
print(recon)

