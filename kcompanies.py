
# coding: utf-8

# In[26]:

import numpy as np
import math
from random import random
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats.stats import pearsonr
from sklearn import linear_model

iteration = 10000
# ordered k companies 
def runk(n, ds):
    secretaries = np.random.rand(n)
    optimal = max(secretaries)
    ranks = scipy.stats.rankdata(secretaries, 'min')
    accepted = []
    #for i in range(1, n):
    #    ranks[i] = (sum(secretaries[:i] < secretaries[i]) + 1) / float(i+1)
    for d in ds:
        sample_n = int(n/d)
        rest_n = n - sample_n
        rest_secretaries = secretaries[sample_n:]
        threshold = max(secretaries[:sample_n]) # max score of the first sample_n secretaries    
        accepts = np.where(rest_secretaries>threshold)[0]
        accepts = [x for x in accepts if x+sample_n not in accepted]
        if len(accepts) != 0:
            accepted.append(accepts[0]+sample_n)
        else:
            for i in range(len(ds)):
                if n-1-i not in accepted:
                    accepted.append(n-1-i)
                    break
    #print threshold, accepted, [secretaries[t] for t in accepted]
    return [ranks[j]==n for j in accepted]

# In[24]:

dss = []
meanss = []
def k_best_d(n, k):
    best_ds = [2.7]
    for i in range(k-1):
        #print i
        d = 1.5
        ds = [d]
        step = .2 + i/20.
        count = 0
        best_mean = 0  
        best_d = 1.5
        means = []
        while count<15 or best_d<3:
            if d>n/15.:
                break
            result = np.array([runk(n, best_ds+[d]) for it in range(iteration)])
            mean = result.mean(axis=0)[i+1]
            #print mean
            if mean > best_mean:
                best_d = d
                count = 1
                best_mean = mean
            else:
                count += 1
            ds.append(d)
            means.append(mean)
            d += step
        best_ds.append(best_d)
        dss.append(ds)
        meanss.append(means)
    return best_ds
        


# In[16]:

print k_best_d(300, 2)


## In[20]:
#
#print meanss[2]
#print dss[2]
#
#
## In[25]:
#
#for n in [100, 500, 1000]:
#    print k_best_d(n, 5)
#
