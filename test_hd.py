import hamming_distance as hd
from g_prior import n_prior
import simple_median as sm
import new_distance as nd
import numpy as np
import eig_algm as ea

# generating data #####################################################
c = 0.9999
p = 12
cov = hd.genCov(p, [[0,1,2,3],[4,5,6,7],[8,9,10,11]], c)
mean = np.zeros(p)
n = 100
sd = 0.1 # sd of error term

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

# find median model ##########################################################
# print("valid covariance matrix: ", hd.isPosDef(cov))
 
# g-prior median model
# print(sm.findMedianSimple(data=data, test_data=testdata, sd=sd))

# testing new distance functions #############################################
# print(nd.findMedian(data, testdata, sd))

# testing eigen value algorithm ##############################################
print(ea.sigmaMinMax(data, hd.unif_prior).shape)