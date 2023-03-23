# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:58:02 2022

@author: Benjamin
"""

# Assignment 2 - Q2
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_pickle('q2data.pkl')

#%% A

base_df = df[:,:,:200]
baserate = np.mean(base_df)*1000

plt.figure()
means = np.mean(base_df,2).reshape(700)*1000
plt.hist(means)
plt.title('Histogram of baseline firing rates (across all trials)')
plt.xlabel('firing rates')

#%% B

def rt_during(df):
    data = df[:,:,200:600]
    mean_rt = np.mean(data, (1,2))*1000
    return mean_rt

def rt_delay(df):
    data = df[:,:,300:700]
    mean_rt = np.mean(data, (1,2))*1000
    return mean_rt

def rt_diff(df):
    data = df[:,:,:600]
    mean_rt =  np.mean(data[:,:,:200], (1,2)) - np.mean(data[:,:,200:600], (1,2))
    return mean_rt*1000

t = np.arange(1,15)

plt.figure()
plt.plot(t, during)
plt.title('Firing Rate During stimulus Presentation per Wavelength')
plt.xlabel('light wavelength #')
plt.ylabel('mean spikes per s')

plt.figure()
plt.plot(t, delay)
plt.title('100ms Delayed Firing Rate per Wavelength')
plt.xlabel('light wavelength #')
plt.ylabel('mean spikes per s')

plt.figure()
plt.plot(t, diff)
plt.title('Firing Rate Difference from Baseline per Wavelength')
plt.xlabel('light wavelength #')
plt.ylabel('change in mean spikes per s')

plt.figure()
plt.plot(t, mean_rt)
plt.title('Firing Rate Percent Change from Baseline per Wavelength')
plt.xlabel('light wavelength #')
plt.ylabel('% change in mean spikes per s')

#%% 
from scipy.stats import norm
data = df[:,:,200:600]
means = np.mean(data,(1,2))*1000
mean, std = norm.fit(means)

plt.plot(t,means/sum(means)-0.13)
x = np.linspace(0, 14, 100)
plt.plot(x, -norm.pdf(x, 8, 5))
plt.show()

#%%

x = np.arange(0,14,0.01)
c = 2
y = 20.32 + (-10)*np.exp(-(x-8)**2 / (2*c**2))


plt.figure()
plt.plot(t, during, label = 'data')
plt.plot(x,y, label = 'model')
plt.title('Gaussian Model fit to Data')
plt.xlabel('light wavelength #')
plt.ylabel('mean spikes per s')
plt.legend()
plt.show()


