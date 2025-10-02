import numpy as np
import scipy as sc
from itertools import product
from sklearn import linear_model
from g_prior import n_prior

# @param data n x p + 1 numpy array of n observations and p features and response
# @param model p-tuple of booleans of what variables to include
def projectionMat(data, model):
    X = data[:, model+(False,)] # we don't want to include response
    temp = np.linalg.inv(X.T @ X)
    return X @ temp @ X.T

def newDist(modelA, modelB, data):
    if sum(modelA) == 0 or sum(modelB) == 0:
        x = 0 # NOTE: is this correct?
    else:
        projA = projectionMat(data, modelA)
        projB = projectionMat(data, modelB)
        x = np.trace(projA @ projB)
    return sum(modelA) + sum(modelB) - 2 * x

# updated find median function for only g-prior #########################
def findMedian(data, test_data, sd):
    # initialising values
    n = data.shape[0]
    p = data.shape[1]-1
    models = list(product([False, True], repeat=p))
    bestModel = None
    bestVal =  None

    for model_hat in models:
        summation = 0
        for model in models:

            # reformatting data
            subdata = data[:, model+(False,)] # we don't want to include response
            y = data[:,p].T
            subtest = test_data[:, model+(False,)]
            testresponse = test_data[:,p].T
            beta = np.zeros(p)

            # # fit mean if null model, otherwise fit a linear model
            # if sum(model) == 0:
            #     mean = np.zeros(n)
            #     new_dist = 1 # i don't know if this is correct
            # else:
            #     reg = linear_model.LinearRegression(fit_intercept=False).fit(subdata, y)
            #     mean = reg.predict(subdata)
            #     beta = reg.coef_
            #     # new distance
            #     new_dist = newDist(model_hat, model, data)
            #     # print("distance: ", new_dist)

            # new distance
            new_dist = newDist(model_hat, model, data)

            # fit mean if null model, otherwise fit a linear model
            if sum(model) == 0:
                mean = np.zeros(n)
            else:
                reg = linear_model.LinearRegression(fit_intercept=False).fit(subdata, y)
                mean = reg.predict(subtest)
                beta = reg.coef_
            
            # calculate the probability of data given linear model
            prob = sc.stats.multivariate_normal.pdf(x=testresponse, mean=mean, cov=sd)
            
            # calculate the probability of data given linear model
            # prob = sc.stats.multivariate_normal.pdf(x=y, mean=mean, cov=sd)
            # print("n prior: ", n_prior(model, beta, sd, data))
            # add to summation
            summation += prob*new_dist*(1/(sum(model)+1))#*n_prior(model, beta, sd, data)

        # storing best model with respect to summation
        if bestVal == None:
            print("sum:", summation)
            print("model:", model_hat)
            bestVal = summation
            bestModel = model_hat
        else:
            if bestVal > summation:
                print(summation)
                bestVal = summation
                bestModel = model_hat
        # print(summation)
    return [bestVal, bestModel]