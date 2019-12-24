import numpy as np

A = np.array([[5, 1], [3, 3]])

eValues, eVectors = np.linalg.eig(A)


eVectors = eVectors.T
print(eVectors)
# print (eValues)

# print(np.dot(A, eVectors[0].T))
# print(np.diag(eValues))
lambdas = np.diag(eValues)
V = eVectors.T

V_inv = np.linalg.inv(V)

# assert A == V.dot(lambdas).dot(V_inv)
print(V.dot(lambdas).dot(V_inv))

