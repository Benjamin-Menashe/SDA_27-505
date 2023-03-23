# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:49:46 2022

@author: Benjamin
"""
#%% Assignment 4 - Q3
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

n = 1000
k = np.concatenate((np.arange(4,2,(-2)/n),np.arange(2,-2-(1/n),(-4)/n)))
sti = np.concatenate((np.arange(0,1,1/n),
                      np.ones(n),
                      np.arange(1,3,(2)/n), 
                      np.arange(3,-1,(-4)/n),
                      np.arange(-1,1,(2)/n),
                      np.arange(1,3,(2)/n),
                      3*np.ones(n),
                      np.arange(3,0,(-3)/n),
                      np.arange(0,-2,(-2)/n),
                      np.arange(-2,-1,(1)/n),
                      np.arange(-1,4,(5)/n),
                      np.arange(4,-3-(1/n),(-7)/n),
                      ))

r_t = np.convolve(sti, k)
r_t = 100+(r_t[:12*n+1]/n)

plt.figure()
plt.title('Estimated r(t) - code estimation')
plt.plot(r_t)
plt.hlines(100, 0, 12*n, color='grey', lw=0.8)
plt.ylabel('Firing Rate')
plt.xlabel('time (ms)')


V = 95 + 10* (1/(1+np.exp(1.2*(100-r_t))))
plt.figure()
plt.title('Estimated r(t) with Sigmoid Non-linearity')
plt.plot(V)
plt.hlines(100, 0, 12*n, color='grey', lw=0.8)
plt.ylabel('Firing Rate')
plt.xlabel('time (ms)')
