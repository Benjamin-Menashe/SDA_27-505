# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:20:02 2022

@author: Benjamin
"""
#%% Assignment 4 - Q1
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(123)

n1_raw = pd.read_csv('psthData1.csv', header=None)
n1 = pd.DataFrame.to_numpy(n1_raw.T)
n2_raw = pd.read_csv('psthData2.csv', header=None)
n2 = pd.DataFrame.to_numpy(n2_raw.T)
stim_raw = pd.read_csv('psthStimOnsets.csv', header=None)
stim = (pd.DataFrame.to_numpy(stim_raw).squeeze())
stim = stim.astype(int)

#%% A

# get peri-stimulus spikes
n1_rast = np.zeros((500,230))
n1_zrast = []
n2_rast = np.zeros((500,230))
n2_zrast = []
for i in range(500):
    n1_rast[i,:] = n1[i,stim[i]-25:stim[i]+205]
    n1_zrast.append(np.flatnonzero(n1_rast[i,5:-5]))
    n2_rast[i,:] = n2[i,stim[i]-25:stim[i]+205]
    n2_zrast.append(np.flatnonzero(n2_rast[i,5:-5]))

# make PSTH and plots    
window = 1/10*np.ones(10)
n1_psth = np.convolve(np.mean(n1_rast,0)*1000, window, 'same')[5:-5]
n2_psth = np.convolve(np.mean(n2_rast,0)*1000, window, 'same')[5:-5]

plt.figure()
plt.suptitle('Neuron 1 - Raster and PSTH')
plt.subplot(2,1,1)
plt.eventplot(n1_zrast)
plt.tick_params(axis='x',bottom=False, labelbottom=False)
plt.ylabel('trial')
plt.vlines(20, 0, 500, color='grey')
plt.subplot(2,1,2)
plt.plot(np.arange(-20,200),n1_psth)
plt.xlabel('time (ms)')
plt.ylabel('Firing Rate')
plt.vlines(0, 40, 120, color='grey')

plt.figure()
plt.suptitle('Neuron 2 - Raster and PSTH')
plt.subplot(2,1,1)
plt.eventplot(n2_zrast)
plt.tick_params(axis='x',bottom=False, labelbottom=False)
plt.ylabel('trial')
plt.vlines(20, 0, 500, color='grey')
plt.subplot(2,1,2)
plt.plot(np.arange(-20,200),n2_psth)
plt.xlabel('time (ms)')
plt.ylabel('Firing Rate')
plt.vlines(0, 40, 120, color='grey')

#%% B

# make JPSTH plots
n1_rast = n1_rast[:,25:125]
n2_rast = n2_rast[:,25:125]

jpsth1 = np.zeros((100,100)) # no reduction
for i in range(500):
    jpsth1 += np.outer(n1_rast[i],n2_rast[i])
jpsth1 = jpsth1/500

plt.figure()
plt.imshow(jpsth1, origin='lower')
plt.xlabel('Neuron 2')
plt.ylabel('Neuron 1')
plt.title('JPSTH - no reduction')


jpsth2 = np.zeros((100,100)) # shift by 1
for i in range(1,500):
    jpsth2 += np.outer(n1_rast[i-1],n2_rast[i])
jpsth2 = jpsth1 - jpsth2/499

plt.figure()
plt.imshow(jpsth2, origin='lower')
plt.xlabel('Neuron 2')
plt.ylabel('Neuron 1')
plt.title('JPSTH - minus shift predictor')


shfl = np.random.permutation(500) # permute
jpsth3 = np.zeros((100,100))
for i in range(500):
    jpsth3 += np.outer(n1_rast[shfl[i]],n2_rast[i])
jpsth3 = jpsth1 - jpsth3/500

plt.figure()
plt.imshow(jpsth3, origin='lower')
plt.xlabel('Neuron 2')
plt.ylabel('Neuron 1')
plt.title('JPSTH - minus stimulus shuffle')
