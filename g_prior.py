import hamming_distance as hd
import numpy as np
import scipy as sc
from sklearn import linear_model

# g prior where g=n, we also assume that the variables are centered
# @param model vector of inclusion indicators
# @param fit fitted beta values from linear regression
# @param sd standard error
# @param data n x p + 1 numpy array of n observations and p features and response
def n_prior(model, fit, sd, data):
    n = data.shape[0]
    p = data.shape[1] - 1
    x = np.zeros(p)
    
    data = data[:, :len(model)]
    A = np.matmul(data.T, data)
    A = np.linalg.inv(A)
    if sum(model) > 1 :

        for i in range(len(model)):
            if model[i]:
                x[i] = fit[sum(model[:i+1])-1]

        prior_prob = sc.stats.multivariate_normal.pdf(x=x, mean=np.zeros(p), cov=n*sd**2*A)
    else:
        prior_prob = sc.stats.multivariate_normal.pdf(x=x, mean=np.zeros(p), cov=n*sd**2*A)
    return prior_prob

# updated find median function for only g-prior #########################
def findMedian(data, test_data, sd):
    # initialising values
    n = data.shape[0]
    p = data.shape[1]-1
    models = list(hd.product([False, True], repeat=p))
    bestModel = None
    bestVal =  None

    for model_hat in models:
        summation = 0
        for model in models:
            # hamming distance
            ham_dist = np.count_nonzero(model_hat!=model)

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
            summation += prob*ham_dist*n_prior(model, beta, sd, data)

        # storing best model with respect to summation
        if bestVal == None:
            print(summation)
            bestVal = summation
            bestModel = model_hat
        else:
            if bestVal > summation:
                print(summation)
                bestVal = summation
                bestModel = model_hat

    return [bestVal, bestModel]
