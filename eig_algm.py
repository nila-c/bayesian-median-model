import numpy as np
from itertools import combinations, product
import new_distance as nd
import copy

# idea: just look at eigen values, see if that's any faster than looking at all matrices
def projectionMat(data, model):
    X = data[:, model+[False]] # we don't want to include response
    temp = np.linalg.inv(X.T @ X)
    return X @ temp @ X.T

# calculates average projection matrix with respect to prior
def avgProjMat(data, prior, *parameters):
    n = data.shape[0]
    p = data.shape[1]-1

    models = list(product([False, True], repeat=p))
    avg_mat = np.zeros((n,n))

    for model in models:
        proj_mat = nd.projectionMat(data, model)
        avg_mat += prior(model, *parameters)*proj_mat

    return avg_mat

# checks if all non-zero eigenvalues are greater than half
# @param mat a square matrix
# @return boolean
def eigHalf(mat, model):
    eig = np.linalg.eigvals(mat)
    eig = np.sort(eig)[-sum(model):]
    return all(eig >= 0.5)

# returns all subsets of variables that meets the eigenvalue criterion
def subsets(indices, length, data, avg_mat):
    subsets = [[False]*length]
    for i in indices:
        subsets_copy = copy.deepcopy(subsets)

        for subset in subsets_copy:
            subset[i] = True
            if cond(subset, data, avg_mat):
              subsets += [subset]
    return subsets[1:]

# checks sigma_min > 0.5
def cond(subset, data, avg_mat):
   mat = projectionMat(data, subset) @ avg_mat
   return eigHalf(mat, subset)

def sigmaMinMax(data, prior, *parameters):
    # initialising values
    p = data.shape[1]-1
    remain_var = []
    possible_models = []

    avg_mat = avgProjMat(data, prior, *parameters)

    for i in range(p):
        model = [False] * p
        model[i] = True
        proj_mat = projectionMat(data, model)
        if eigHalf(proj_mat @ avg_mat, model):
            remain_var += [i]
    
    subsets = np.array(subsets(remain_var, p, data, avg_mat))
    rank = np.sum(subsets, axis=1)
    max_rank = np.max(rank)

    for i in range(len(subsets)):
        if rank[i] == max_rank:
            possible_models += [subsets[i,:]]

    possible_models = np.array(possible_models)
    possible_models = np.unique(possible_models, axis=0)

    return possible_models

# write code to check if only 1 signal per cluster