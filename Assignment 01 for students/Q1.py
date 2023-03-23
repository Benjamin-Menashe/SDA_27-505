# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 11:06:35 2022

@author: Benjamin
"""

# SDA_2022 Assignment 01
import numpy as np
import scipy as sig
import matplotlib.pyplot as plt
np.random.seed(123)

#%% Q1a

EEG_t = np.arange(start = 0, stop = 4, step = 0.010)
EEG_wake = np.cos(EEG_t[:200]*24*np.pi)
EEG_one = np.cos(EEG_t[200:]*12*np.pi)
EEG_all = np.concatenate((EEG_wake, EEG_one))

plt.figure
plt.scatter(EEG_t, EEG_all, marker = '.')
plt.title('Human EEG During Sleep')
plt.ylabel('µV')
plt.xlabel('time (s)')
plt.show()

Mouse_t = np.arange(start = 0, stop = 4, step = 0.001)
Mouse_wake = np.cos(Mouse_t[:2000]*24*np.pi)
Mouse_one = np.cos(Mouse_t[2000:]*12*np.pi)
Mouse_all = np.concatenate((Mouse_wake, Mouse_one))

plt.figure
plt.scatter(Mouse_t, Mouse_all, marker = '.')
plt.title('Mouse Extracellular Recordings in the Hippocampus Region During Sleep')
plt.ylabel('µV')
plt.xlabel('time (s)')
plt.show()

#%% Q1b
# 40 ms bins
bin_4_t = np.arange(start = 0.015, stop = 3.985, step = 0.040)
bin_4_val = np.zeros(100)
for i in np.arange(0,100):
    bin_4_val[i] = np.mean(EEG_all[4*i:4*i+4])
    
bin_5_t = np.arange(start = 0.020, stop = 3.980, step = 0.050)
bin_5_val = np.zeros(80)
for i in np.arange(0,80):
    bin_5_val[i] = np.mean(EEG_all[5*i:5*i+5])
    
plt.figure
plt.suptitle('Human EEG During Sleep')
plt.title('bin size = 4', fontsize = 10)
plt.plot(bin_4_t, bin_4_val, '.-')
plt.ylabel('µV')
plt.xlabel('time (s)')
plt.show()

plt.figure
plt.suptitle('Human EEG During Sleep')
plt.title('bin size = 5', fontsize = 10)
plt.plot(bin_5_t, bin_5_val, '.-')
plt.ylabel('µV')
plt.xlabel('time (s)')
plt.show()

#%% Q1c
alias_t = np.arange(start = 0, stop = 4, step = 1/9)
alias_wake = np.cos(alias_t[:18]*24*np.pi)
alias_one = np.cos(alias_t[18:]*12*np.pi)
alias_all = np.concatenate((alias_wake, alias_one))

plt.figure
plt.title('Undersampling Can Alias Sleep Phases')
plt.plot(Mouse_t, Mouse_all, '-', linewidth = 0.5, label = '1000 Hz')
plt.plot(alias_t, alias_all, '.-', label = '9 Hz')
plt.ylabel('µV')
plt.xlabel('time (s)')
plt.legend(loc=4)
plt.show()
