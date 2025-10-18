import numpy as np
import scipy as sc
from itertools import permutations, product
from sklearn import linear_model

# create synthetic example of poor hamming distance result ############

# gets p from model
# @param model partition expressed as an array of arrays
def findP(model):
    i = []
    for cluster in model:
        i += [max(cluster)]
    return(max(i)+1)

# creates covariance matrix as seen above wrt a particular model
# @param size int, n
# @param model partition expressed as an array of arrays
# @param c covariance within signal cluster
# @return a numpy matrix/array with 1's along diagonal and covariance c within clusters and 0 elsewhere
def genCov(size, model, c):
    x = np.zeros(shape=(size,size))
    for part in model:
        for j,k in permutations(part, r=2):
            x[j,k] = c
    for i in range(size):
        x[i,i] = 1
    return x

# check if positive definite
# @param numpy matrix
def isPosDef(x):
    return np.all(np.linalg.eigvals(x) > 0)

# partitions some set of elements into its various partitions
# @param collection of elements in an array to partition
# @return iter object of partitions
def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

# priors ##############################################################

# @param model vector of inclusion indicators
def unif_prior(model):
    denom = 2**len(model)
    return 1/denom

def geom_prior(model):
    return 0.5**sum(model)

# finding median model ################################################
# with specfied prior
def findMedian(data, test_data, prior, sd):
    # initialising values
    n = data.shape[0]
    p = data.shape[1]-1
    models = list(product([False, True], repeat=p))
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

            # fit mean if null model, otherwise fit a linear model
            if sum(model) == 0:
                mean = np.full(fill_value=np.mean(y), shape=n)
            else:
                reg = linear_model.LinearRegression().fit(subdata, y)
                mean = reg.predict(subtest)
            
            # calculate the probability of data given linear model
            prob = sc.stats.multivariate_normal.pdf(x=testresponse, mean=mean, cov=sd)

            # add to summation
            summation += prob*ham_dist*prior(model)

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

