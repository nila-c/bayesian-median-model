import numpy as np
import scipy as sc
from itertools import permutations, product
from sklearn import linear_model

# g prior where g=n, we also assume that the variables are centered
# @param model vector of inclusion indicators
# @param fit fitted beta values from linear regression
# @param sd standard error
# @param data n x p numpy array of n observations and p features
def n_prior(model, fit, sd, data):
    n = data.shape[0]
    x = np.zeros(n)

    for i in range(len(model)):
        if model[i]:
            x[i] = fit[sum(model[:i+1])-1]

    A = np.matmul(data.T, data)
    A = np.linalg.inv(A)
    return sc.stats.multivariate_normal.pdf(x=x, mean=np.zeros(n), cov=n*sd**2*A)

