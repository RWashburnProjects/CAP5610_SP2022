#CAP5610
#HW4

#####TASK 1. Implement K-Means from scratch
#Select K points as initial centroid & repeat
#Form K clusters by assigning all points to the closest centroid
#until the centroids don't change

###QUESTION 1


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance

data = pd.read_csv('/Users/renitawashburn/Downloads/hw4_kmeans_data/data.csv',header=None)
label = pd.read_csv('/Users/renitawashburn/Downloads/hw4_kmeans_data/label.csv',header=None)

X = data.values
y= label

#Euclidean
def kmeans(X, k=10, max_iterations=100):
    if isinstance(X, pd.DataFrame):X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(distance.cdist(X, centroids, 'euclidean'), axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[P==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'euclidean'),axis=1)
        if np.array_equal(P, tmp):break
        # QUESTION 3
        #print(bool(np.array_equal(P, tmp)))
        P = tmp
    return P

P = kmeans(X)

from sklearn.metrics import mean_squared_error
print("Euclidean SSE: ",mean_squared_error(y, P))
#QUESTION 3
print(bool(np.array_equal(P, tmp)))

#Cosine
def kmeans(X, k=10, max_iterations=100):
    if isinstance(X, pd.DataFrame):X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    Q = np.argmin(distance.cdist(X, centroids, 'cosine'), axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[Q==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'cosine'),axis=1)
        if np.array_equal(Q, tmp):break
        # QUESTION 3
        #print(bool(np.array_equal(P, tmp)))

        Q = tmp
    return Q

Q = kmeans(X)

from sklearn.metrics import mean_squared_error
print("Cosine SSE: ",mean_squared_error(y, Q))


#Jaccard
def kmeans(X, k=10, max_iterations=100):
    if isinstance(X, pd.DataFrame):X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    R = np.argmin(distance.cdist(X, centroids, 'jaccard'), axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[R==i,:].mean(axis=0) for i in range(k)])
        tmp = np.argmin(distance.cdist(X, centroids, 'jaccard'),axis=1)
        if np.array_equal(R, tmp):break
        R = tmp
    return R

R = kmeans(X)

from sklearn.metrics import mean_squared_error
print("Jaccard SSE: ",mean_squared_error(y, R))

#QUESTION 2
from sklearn.metrics import accuracy_score

print("Euclidean accuracy: ",accuracy_score(y, P))
print("Cosine accuracy: ",accuracy_score(y, Q))
print("Jaccard accuracy: ",accuracy_score(y, R))


#QUESTION 3 & 4
#Change distance type to output conditions
def kmeans(X, k=10, max_iterations=100):
    if isinstance(X, pd.DataFrame):X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    R = np.argmin(distance.cdist(X, centroids, 'jaccard'), axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[R==i,:].mean(axis=0) for i in range(k)])
        print("centroid ", centroids)
        tmp = np.argmin(distance.cdist(X, centroids, 'jaccard'),axis=1)
        print("tmp ", tmp)
        if np.array_equal(R, tmp):break
        R = tmp
    return R

R = kmeans(X)

from sklearn.metrics import mean_squared_error
print("SSE: ",mean_squared_error(y, R))



#####TASK 2. Calculate various euclidean distances
import numpy as np
from scipy.spatial import distance_matrix
a = np.array([[4.7,3.2],[4.9,3.1],[5,3],[4.6,2.9]])
b = np.array([[5.9,3.2],[6.7,3.1],[6,3],[6.2,2.8]])

# Display the matrices
print("matrix x:\n", a)
print("matrix y:\n", b)

# compute the Euclidean distance (p=2) matrix
dist_mat = distance_matrix(a, b, p=2)

# display distance matrix
print("Distance Matrix:\n", dist_mat)
