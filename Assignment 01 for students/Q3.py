# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:57:05 2022

@author: Benjamin
"""
# SDA_2022 Assignment 01
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

np.random.seed(123)

# Q2

def StochProperties(mat):
    t = np.arange(0, 5, 0.01)
    is_stat = True
    is_ergod = True
    
    # Stationary: check consistency of means and stds for each 5 s trials
    trial_mean = np.mean(mat,1)
    trial_std = np.std(mat,1)
    
    if np.std(trial_mean) > 0.05 or np.std(trial_std) > 0.05:
        is_stat = False
        is_ergod = False
    
    # Ergodic: check mean and std for each 500 timepoints
    tp_mean = np.mean(mat,0)
    tp_std = np.std(mat,0)
    if np.std(tp_mean) > 0.05 or np.std(tp_std) > 0.05:
        is_ergod = False
    
    return is_stat, is_ergod


trial_mean = np.mean(mat,1)
trial_std = np.std(mat,1)

plt.figure
plt.plot(trial_mean, label = 'Means')
plt.plot(trial_std, label = 'Stds')
plt.legend()
plt.title('EEG3 - mean and std for each trial')
plt.xlabel('trial number')
plt.show

plt.figure
for i in range(200):
    cdf = ECDF(mat[i,:])
    plt.plot(cdf.x, cdf.y)
    plt.title('EEG3 - Empirical CDF for each trial across 5 s')
plt.xlabel('X')
plt.ylabel('CP(X)')
plt.show()


tp_mean = np.mean(mat,0)
tp_std = np.std(mat,0)

plt.figure
plt.plot(tp_mean, label = 'Means')
plt.plot(tp_std, label = 'Stds')
plt.legend()
plt.title('EEG3 - mean and std for each timepoint')
plt.xlabel('time (s)')
plt.show

plt.figure
for i in range(500):
    cdf = ECDF(mat[:,i])
    plt.plot(cdf.x, cdf.y)
    plt.title('EEG3 - Empirical CDF for each timepoint across trials')
plt.xlabel('X')
plt.ylabel('CP(X)')
plt.show()

plt.figure
plt.imshow(mat)
plt.title('EEG3 - colorplot')
plt.xlabel('time (s)')
plt.ylabel('trial number')
plt.colorbar(shrink = 0.5)
plt.show()

    
