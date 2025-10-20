import sys
from pathlib import Path
from pipe import select, where, chain

# 1. Get the path to the current file (new_dist_plot.py)
current_file_path = Path(__file__)

# 2. Get the path to the 'bayesian_median_model' directory
lib_dir = current_file_path.parent.parent

# 3. We convert the Path object to a string
sys.path.append(str(lib_dir))

# Now you can import library as if it were in the same folder
import new_distance as nd
import hamming_distance as hd
import simple_median as sm
import numpy as np

def genData(mean, cov, n, sd):
        # generating variable data
        data = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

        # calculating response
        epsilon = np.random.normal(0, scale=sd, size=n)
        epsilon = np.atleast_2d(epsilon).T
        # NOTE: will assume epsilon sd is known

        temp = np.append(data, epsilon, axis=1).sum(axis=1)
        temp = np.atleast_2d(temp).T
        data = np.append(data, temp, axis=1)
        return data

# checking if the model aquired picks all signals (doesn't check if multiple from same signal are there)
def goodModel(model, vars_selected):
    vars_selected = np.array(vars_selected).nonzero()[0]
    clusters = []
    
    for chunk in model:
        clusters += [bool(set(chunk) & set(vars_selected))]

    return all(clusters)

def difCovData(model, cor_range, evals, n, sd):
    # creating range for correlation
    times = range(0, evals)
    start = cor_range[0]
    end = cor_range[1]
    cor = np.multiply(((end-start)/evals), times)
    cor = np.add(cor, start)

    # initialising values
    p = hd.findP(model)
    mean = np.zeros(p)
    df = np.zeros(shape=(evals,3))
    models = []

    # caluclate model for every correlation c and see if it has all signals
    for c in cor:
        # generate data with specified correlation
        i = np.where(cor == c)
        df[i,0] = c
        cov = hd.genCov(p, model, c)
        data = genData(mean, cov, n, sd)
        testdata = genData(mean, cov, n, sd)

        # find new distance model
        new_dist_final = nd.findMedian(data, testdata, sd)
        models += [new_dist_final[1]]
        val = goodModel(model, new_dist_final[1])
        df[i,1] = int(val)

        # find simple median model
        median_final = sm.findMedianSimple(data, testdata, sd)
        val = goodModel(model, median_final[1])
        df[i,2] = int(val)

    return [df, models]

model = [[0,1,2,3],[4,5,6,7]]
cor_range = [0.01,1]
evals = 10
n = 50
sd = 0.1
print(difCovData(model, cor_range, evals, n, sd))