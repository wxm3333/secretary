
# coding: utf-8

# In[26]:

import numpy as np
import math
import time as tm
from random import random
import scipy.stats
from scipy.stats.stats import pearsonr

iteration = 50
def runpk(n, ds, ps):
    secretaries = np.random.rand(n)
    ranks = scipy.stats.rankdata(secretaries, 'min')
    accepted = []
    #for i in range(1, n):
    #    ranks[i] = (sum(secretaries[:i] < secretaries[i]) + 1) / float(i+1)
    for i in range(len(ds)):
        d = ds[i]
        p = ps[i]
        sample_n = int(n/d)
        rest_n = n - sample_n
        rest_secretaries = secretaries[sample_n:]
        threshold = max(secretaries[:sample_n]) # max score of the first sample_n secretaries
        p_random = np.random.rand(rest_n) # random numbers between 0 and 1
        accepts = np.where((p>p_random) & (rest_secretaries>threshold))[0]
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



ds = np.arange(1.1, 20, .1)
ps = np.arange(.02, .98, .05)

def best_prob_1(n):
    mean1 = []
    mean2 = []
    for pp1 in ps:
	for d1 in ds:
	    result = np.array([runpk(n, [d1], [pp1]) for i in range(iteration)])
	    mean = result.mean(axis=0)
	    mean2.append(mean[0])
    b = np.array(mean2).reshape(len(ps), len(ds))
    mean_file = open('n_'+str(n)+'_prob.txt', 'w')
    mean_file.write('p,'+','.join([str(x) for x in ds])+'\n')
    for i in range(len(ps)):
	mean_file.write(str(ps[i])+','+','.join([str(x) for x in b[i]])+'\n')
    mean_file.close()
    b_best = [round(ds[np.argmax(x)],1) for x in b]
    return b_best


# In[16]:
#out = open('out_prob.txt','w')
#out.write('n,'+'.'.join([str(x) for x in ps])+'\n')
ns = [50, 100, 300, 500, 1000]
#for n in ns:
#    t = tm.time()
#    result = best_prob_1(n)
#    time = round((tm.time() - t)/60.,1)
#    print result
#    out.write(','.join(str(x) for x in [n]+result+[time])+'\n')
#out.close()

read = open('out_prob.txt','r')
read.readline()
d1s = {}
for line in read:
    data = line.split(',')
    n = int(data[0])
    d1s[n] = [float(x) for x in data[1:]]

pp1s = np.arange(.52, 1, .1)
for n in ns[1:]:
    mean2 = []
    out = open('n_'+str(n)+'_2prob'+'.txt', 'w')
    out.write('pp1,'+','.join([str(x) for x in np.arange(.05, 1, .05)])+'\n')
    for pp1 in pp1s:
	i = ps.tolist().index(pp1)
	pp2s = np.arange(.05, pp1, .05)
	mean_file = open('n_'+str(n)+'_pp1_'+str(pp1)+'.txt', 'w')
	mean_file.write('p,'+','.join([str(x) for x in ds])+'\n')
	for pp2 in pp2s:
	    d1 = d1s[n][i]
	    for d2 in ds:
		result = np.array([runpk(n, [d1, d2], [pp1, pp2]) for i in range(iteration)])
		mean = result.mean(axis=0)
		mean2.append(mean[1])
	b = np.array(mean2).reshape(len(pp2s), len(ds))
	for i in range(len(pp2s)):
	    mean_file.write(str(pp2s[i])+','+','.join([str(x) for x in b[i]])+'\n')
	mean_file.close()
	b_best = [round(ds[np.argmax(x)],1) for x in b]
	out.write(','.join(str(x) for x in [pp1]+b_best)+'\n')
	print b_best

    out.close()
