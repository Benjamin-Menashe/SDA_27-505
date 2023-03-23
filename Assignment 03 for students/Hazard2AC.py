# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:47:06 2022

@author: Benjamin
"""
import numpy as np
import matplotlib.pyplot as plt


def Hazard2Autocorr(haz):
    ac = np.zeros(len(haz))
    surv = np.exp(-(np.cumsum(haz)))
    
    for i in range(len(haz)):
        ac[i] += np.exp(np.log(surv[i-1]) + np.log(haz[i]))
        for j in range(i):
            ac[i] += np.exp(np.log(surv[j-1]) + np.log(haz[j]) + np.log(surv[i-j-1]) + np.log(haz[i-j]))
            for k in range(j):
                ac[i] += np.exp(np.log(surv[k-1]) + np.log(haz[k]) + np.log(surv[j-k-1]) + np.log(haz[j-k]) + np.log(surv[i-j-1]) + np.log(haz[i-j]))
    
    ac[0] = 0
    ac2 = np.zeros(2*len(haz)-1)
    ac2[len(haz)-1:] = ac
    ac2[:len(haz)] = np.flip(ac)
    return ac2

haz = np.array([0, 0, 0.18, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05])
haz2 = np.zeros(10) + 0.05
haz = np.concatenate((haz, haz2))

t = np.arange(-100, 105, 5)

ac = Hazard2Autocorr(haz)

plt.figure()
plt.plot(t, ac)
plt.title('Autocorrelation')
plt.xlabel('time after spike (ms)')
plt.ylabel('Approximate P(firing)')
plt.show()
