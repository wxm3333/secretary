
# coding: utf-8

# In[26]:

import numpy as np
import math
import time as tm
from random import random
import scipy.stats
from scipy.stats.stats import pearsonr

iteration = 50000
# ordered k companies 
def runk(n, ds):
    secretaries = np.random.rand(n)
    optimal = max(secretaries)
    ranks = scipy.stats.rankdata(secretaries, 'min')
    accepted = []
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
out = open('out.txt','w')
dss = np.arange(1.1, 100, .1)
def k_best_d(n, k):
    means_file = open('n_'+str(n)+'.txt','w')
    best_ds = []
    for i in range(k):
        means = []
        for d in dss:
            if (d>15 and d>n/10.) or d>30:
                break
            result = np.array([runk(n, best_ds+[d]) for it in range(iteration)])
            mean = result.mean(axis=0)[i]
            means.append(mean)
	if i == 0:
	    means_file.write(','.join(str(x) for x in dss[:len(means)])+'\n')
	means_file.write(','.join(str(x) for x in means)+'\n')
	best_d = dss[np.argmax(means)]
        best_ds.append(round(best_d, 1))
    return best_ds
        


# In[16]:
ns = [50, 100, 300, 500, 1000]
for n in ns:
    t = tm.time()
    result = k_best_d(n, 10)
    time = round((tm.time() - t)/60.,1)
    print result
    out.write(','.join(str(x) for x in [n]+result+[time])+'\n')
out.close()

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
