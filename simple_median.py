import numpy as np
import scipy as sc
from itertools import permutations, product
from sklearn import linear_model
import hamming_distance as hd
import g_prior as gp

# updated find median function for only g-prior #########################
def findMedianSimple(data, test_data, sd):
    # initialising values
    n = data.shape[0]
    p = data.shape[1]-1
    models = list(hd.product([False, True], repeat=p))
    bestModel = [False] * p
    bestVal =  None

    for i in range(p):
        summation = 0
        for model in models:
            # reformatting data
            subdata = data[:, model+(False,)] # we don't want to include response
            y = data[:,p].T
            subtest = test_data[:, model+(False,)]
            testresponse = test_data[:,p].T
            beta = np.zeros(p)

            # fit mean if null model, otherwise fit a linear model
            if sum(model) == 0:
                mean = np.zeros(n)
            else:
                reg = linear_model.LinearRegression(fit_intercept=False).fit(subdata, y)
                mean = reg.predict(subtest)
                beta = reg.coef_
            
            # calculate the probability of data given linear model
            prob = sc.stats.multivariate_normal.pdf(x=testresponse, mean=mean, cov=sd)

            # add to summation
            summation += prob*gp.n_prior(model, beta, sd, data)
        print(summation)

        # storing best model with respect to summation
        if summation > 0.5:
            print(summation)
            bestVal = summation
            bestModel[i] = True

    return [bestVal, bestModel]
