import numpy as np
import scipy as sc
from itertools import permutations, product
from sklearn import linear_model

# create synthetic example of poor hamming distance result ############

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

# generating data #####################################################
c = 0.9999
p = 8
cov = genCov(p, [[0,1,2,3,4,5,6,7]], c)
mean = np.zeros(p)
n = 100
sd = 0.1 # sd of error term

# check if positive definite
# @param numpy matrix
def isPosDef(x):
    return np.all(np.linalg.eigvals(x) > 0)

print("valid covariance matrix: ", isPosDef(cov))

# generating variable data
data = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

# calculating response
epsilon = np.random.normal(0, scale=sd, size=n)
epsilon = np.atleast_2d(epsilon).T
# NOTE: will assume epsilon sd is known

temp = np.append(data, epsilon, axis=1).sum(axis=1)
temp = np.atleast_2d(temp).T
data = np.append(data, temp, axis=1)

# testing data
testdata = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
epsilon = np.random.normal(0, scale=sd, size=n)
epsilon = np.atleast_2d(epsilon).T
temp = np.append(testdata, epsilon, axis=1).sum(axis=1)
temp = np.atleast_2d(temp).T
testdata = np.append(testdata, temp, axis=1)

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
    return 1

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
            bestVal = summation
        else:
            if bestVal > summation:
                print(summation)
                bestVal = summation
                bestModel = model_hat

    return [bestVal, bestModel]

print(findMedian(data=data, test_data=testdata, prior=unif_prior, sd=sd))