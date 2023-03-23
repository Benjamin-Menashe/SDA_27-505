# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:04:35 2022

@author: Benjamin
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

n = 5000000
A = np.zeros(n)
A[np.arange(100, n, 300)] = 1
A[np.arange(300, n ,300)] = 1

t = np.arange(-500,500+1)

A_acs = np.zeros(1000+1)
for i in range(1,500+1):
    A_acs[500+i] = np.sum(np.logical_and(A[:-i], A[i:]))/np.sum(A[:-i])
A_acs[500] = 0
A_acs[:500] = np.flip(A_acs[500+1:])
A_acs = A_acs*1000

plt.figure()
plt.title('Autocorrelation - Neuron A')
plt.plot(t, A_acs)
plt.xlabel('time after spike (ms)')
plt.ylabel('Firing Rate (hz)')
plt.show()

B = np.zeros(n)
for i in range(24, n):
    p = 20/1000 if np.any(A[i-24:i-4]) else 10/1000
    B[i] = np.random.binomial(1,p)
    
B_acs = np.zeros(1000+1)
for i in range(1,500+1):
    B_acs[500+i] = np.sum(np.logical_and(B[:-i], B[i:]))/np.sum(B[:-i])
B_acs[500] = 0
B_acs[:500] = np.flip(B_acs[500+1:])
B_acs = B_acs*1000

plt.figure()
plt.title('Autocorrelation - Neuron B')
plt.plot(t, B_acs)
plt.xlabel('time after spike (ms)')
plt.ylabel('Firing Rate (hz)')
plt.show()

AB_CC = np.zeros(1000+1)
for i in range(-500,500):
    AB_CC[500+i] = np.sum(np.logical_and(A[500:-500], B[i+500:-500+i]))/np.sum(A[500:-500])

AB_CC = AB_CC*1000
AB_CC[-1] =  AB_CC[-2]

plt.figure()
plt.title('Cross-correlation (A, B)')
plt.plot(t, AB_CC)
plt.xlabel('time after spike (ms)')
plt.ylabel('Firing Rate (hz)')
plt.show()
   