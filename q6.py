"""
@author: Akdeniz Kutay Ocal
Title: CMPE442_TakeHome_Q6
Description: Question 6 - PCA
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def PCA(X):
    """
    Reduces the dimensions of feature set to 2
    :param X: np.array
    :return: np.array
    """

    # Centering X
    X = X - X.mean(axis=0)

    # Data matrix X, assumes 0-centered
    n, m = X.shape

    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)

    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)

    pc1_value = float('-inf')
    pc2_value = float('-inf')
    pc1_index = float('-inf')
    pc2_index = float('-inf')

    for i in range(len(eigen_vals)):
        if eigen_vals[i] > pc1_value:
            pc2_value = pc1_value
            pc2_index = pc1_index
            pc1_value = eigen_vals[i]
            pc1_index = i

        elif eigen_vals[i] > pc2_value:
            pc2_value = eigen_vals[i]
            pc2_index = i

    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)

    pcaList = np.zeros((2,1797))
    pcaList[0] = (X_pca[:, pc1_index])
    pcaList[1] = (X_pca[:, pc2_index])

    return pcaList.T

# Loading the data
digits = datasets.load_digits()
X = digits.data

# converting PCA so that it is same as the given graph
x_pca = - PCA(X)

pc1 = []
pc2 = []
colors = ["red", "green", "blue", "cyan", "purple"]

# First Five Digits

"""
for j in range(5):
    for i in range(len(digits.data)):
        if digits.target_names[digits.target[i]] == j:
            pc1.append(x_pca[i][0])
            pc2.append(x_pca[i][1])
    plt.plot(pc1, pc2, "*", color=colors[j])
    pc2 = []
    pc1 = []

"""
# Last Five Digits

"""for j in range(5):
    for i in range(len(digits.data)):
        if digits.target_names[digits.target[i]] == 5 + j:
            pc1.append(x_pca[i][0])
            pc2.append(x_pca[i][1])
    plt.plot(pc1, pc2, "*", color=colors[j])
    pc2 = []
    pc1 = []

"""

# Even Digits

"""
for j in range(5):
    for i in range(len(digits.data)):
        if digits.target_names[digits.target[i]] == 2 * j:
            pc1.append(x_pca[i][0])
            pc2.append(x_pca[i][1])
    plt.plot(pc1, pc2, "*", color=colors[j])
    pc2 = []
    pc1 = []

"""
# Odd Digits
"""
for j in range(5):
    for i in range(len(digits.data)):
        if digits.target_names[digits.target[i]] == 2 * j + 1:
            pc1.append(x_pca[i][0])
            pc2.append(x_pca[i][1])
    plt.plot(pc1, pc2, "*", color=colors[j])
    pc2 = []
    pc1 = []
"""

# 0,3,6,9

"""
for j in range(5):
    for i in range(len(digits.data)):
        if digits.target_names[digits.target[i]] == 3 * j:
            pc1.append(x_pca[i][0])
            pc2.append(x_pca[i][1])
    plt.plot(pc1, pc2, "*", color=colors[j])
    pc2 = []
    pc1 = []
"""

plt.show()
