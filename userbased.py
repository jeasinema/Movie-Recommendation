from __future__ import division
import numpy as np
from scipy import linalg
from numpy import dot
from scipy.sparse import csc_matrix
import os
from numpy.random import random
from random import shuffle
from sklearn.decomposition import ProjectedGradientNMF
from scipy.spatial.distance import cosine, correlation, euclidean
from scipy.stats import pearsonr


def similarity(x, y):
    idx = filter(lambda i: x[i] != 0 and y[i] != 0, range(x.shape[0]))
    return euclidean(x[idx], y[idx])


def predict(X, x):
    sim = np.array([similarity(X[i], x) for i in range(X.shape[0])])
    sim = (np.max(sim) - sim) / np.max(sim)
    return np.dot(sim, X) / np.sum(sim)

X = np.array([[1, 2, 5, 5], [5, 5, 1, 1], [1, 1, 4, 5]])
x = np.array([1, 1, 0, 0])
print predict(X, x)
# print correlation([1, 2, 2, 1], [3, 3, 1, 6])
# print similarity(X[0], x)
