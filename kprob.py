
# coding: utf-8

# In[26]:

import numpy as np
import math
import time as tm
from random import random
import scipy.stats
from scipy.stats.stats import pearsonr

iteration = 50000
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



ds = np.arange(1.1, 30, .1)
ps = np.arange(.05, 1, .05)
ns = [50, 100, 300, 500, 1000]
#group_ps = [.8, .6, .4, .2]
group_ps = np.arange(.9, 0, -.1)

#out = open('k_prob_out4.txt', 'w')
#out2 = open('k_prob_out_mean4.txt', 'w')
#for n in ns[1:]:
#    best_ds = []
#    best_means = []
#    means_file = open('n_'+str(n)+'_group_'+str(9)+'.txt','w')
#    means_file.write('prob,'+','.join(str(x) for x in ds)+'\n')
#    for i in range(1,len(group_ps)+1):
#	probs = group_ps[:i]
#	means = []
#	for d in ds:
#	    result = np.array([runpk(n, best_ds+[d], probs) for ite in range(iteration)])
#	    mean = result.mean(axis=0)[-1]
#	    means.append(mean)
#	means_file.write(','.join(str(x) for x in [probs[-1]]+means)+'\n')
#	best_means.append(max(means))
#	best_ds.append(round(ds[np.argmax(means)],1))
#    means_file.close()
#    out.write(','.join(str(x) for x in [n]+best_ds)+'\n')
#    out2.write(','.join(str(x) for x in [n]+best_means)+'\n')
#    print best_ds
#out.close()
#out2.close()

def k_best(n, accept_probs):
    best_means = []
    best_ds=[]
    means_file = open('n_'+str(n)+'_kprob_'+';'.join([str(x) for x in accept_probs])+'.txt','w')
    means_file.write('prob,'+','.join(str(x) for x in ds)+'\n')
    for i in range(len(best_ds)+1,len(accept_probs)+1):
	probs = accept_probs[:i]
	means = []
	for d in ds:
	    result = np.array([runpk(n, best_ds+[d], probs) for ite in range(iteration)])
	    mean = result.mean(axis=0)[-1]
	    means.append(mean)
	means_file.write(','.join(str(x) for x in [probs[-1]]+means)+'\n')
	best_means.append(max(means))
	best_ds.append(round(ds[np.argmax(means)],1))
    means_file.close()
    return best_ds, best_means

def collude(n, best_p, rest_p):
    best_ds, best_means = k_best(n, [best_p]+rest_p)
    best_ds2, best_means2 = k_best(n, [sum(rest_p), best_p])
    counts = [0]*(len(rest_p)+1)
    trials = 100000
    for ite in range(trials):
	result = runpk(n, best_ds2, [sum(rest_p), best_p])
	if result[1]:
	    counts[-1] += 1
	elif result[0]:
	    rand = np.random.multinomial(1, rest_p)
	    counts[rand.tolist().index(1)] += 1
    success_probs = np.array(counts)/float(trials)
    best_means = [round(x,4) for x in best_means]
    success_probs = [round(x,4) for x in success_probs]
    return best_means, [success_probs[-1]]+success_probs[:-1], best_ds

best_ps = [.5, .6, .7, .8]
rest_ps = [[.4,.3,.2,.1],[.4,.2,.1],[.4,.15]]
this_n = 300
out = open('collude_'+str(this_n)+'.txt','w')
for best_p in best_ps:
    for rest_p in rest_ps:
	if sum(rest_p)<=best_p or best_p<=max(rest_p):
	    continue
	else:
	    result = collude(this_n, best_p, rest_p)
	    print result
	    for r in result:
		out.write(';'.join([str(x) for x in [best_p]+rest_p])+',')
		out.write(','.join([str(x) for x in r])+'\n')
out.close()
    
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
#for n in ns:
#    t = tm.time()
#    result = best_prob_1(n)
#    time = round((tm.time() - t)/60.,1)
#    print result
#    out.write(','.join(str(x) for x in [n]+result+[time])+'\n')
#out.close()

#read = open('out_prob.txt','r')
#read.readline()
#d1s = {}
#for line in read:
#    data = line.split(',')
#    n = int(data[0])
#    d1s[n] = [float(x) for x in data[1:]]
#
#pp1s = np.arange(.52, 1, .1)

#for n in ns[1:]:
#    out = open('n_'+str(n)+'_2prob'+'.txt', 'w')
#    out.write('pp1,'+','.join([str(x) for x in np.arange(.05, 1, .05)])+'\n')
#    for pp1 in pp1s:
#    	mean2 = []
#	idx = int(10+2*(pp1-.52)/.1)
#	pp2s = np.arange(.05, pp1, .05)
#	mean_file = open('n_'+str(n)+'_pp1_'+str(pp1)+'.txt', 'w')
#	mean_file.write('p,'+','.join([str(x) for x in ds])+'\n')
#	for pp2 in pp2s:
#	    d1 = d1s[n][idx]
#	    for d2 in ds:
#		result = np.array([runpk(n, [d1, d2], [pp1, pp2]) for i in range(iteration)])
#		mean = result.mean(axis=0)
#		mean2.append(mean[1])
#	b = np.array(mean2).reshape(len(pp2s), len(ds))
#	for i in range(len(pp2s)):
#	    mean_file.write(str(pp2s[i])+','+','.join([str(x) for x in b[i]])+'\n')
#	mean_file.close()
#	b_best = [round(ds[np.argmax(x)],1) for x in b]
#	out.write(','.join(str(x) for x in [pp1]+b_best)+'\n')
#	print b_best
#
#    out.close()
