# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:00:43 2022

@author: Benjamin
"""
# Assignment 4 - Q2 B & C
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

## LOAD 'RESP.MAT' AND 'STIM.MAT' MANUALLY BEFORE PROCEEDING

# make "PSTH"
r_t_raw = np.mean(resp,0)*1000

# smoothing window and plot
gwindow = sig.windows.gaussian(500,150)
gwindow = gwindow/np.sum(gwindow)
r_t = np.convolve(r_t_raw, gwindow, 'same')

plt.figure()
plt.title('rate function')   
plt.plot(np.arange(0,60,0.001),r_t)
plt.xlabel('time (s)')
plt.ylabel('Firing rate (Hz)')


# calculate kernel using known properties of white-noise stimuli and plot
s_var = np.var(stim)
krnl = np.zeros(200)
for i in range(1,200):
    krnl[i] = np.sum(r_t_raw[i:]*stim[:-i])/(s_var*(60000-i))

plt.figure()
plt.title('Linear Kernel')
plt.plot(np.arange(-199,1), np.flip(krnl))
plt.xlabel('time (ms)')
plt.ylabel('firing rate increase')


# use kernel up to 20ms to estimate neural rate and plot
est_r_t_raw = np.convolve(stim, krnl[:20], 'same')
est_r_t = np.convolve(est_r_t_raw, gwindow, 'same')
est_r_t = np.mean(r_t)+est_r_t

plt.figure()
plt.title('Actual vs. Estimated Neuron Firing Rate (first 20 s)')
plt.plot(r_t[:20000], label='Neural rate')
plt.plot(est_r_t[:20000], label='Estimated rate')
plt.xlabel('time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.legend()


# calculate RMSE and plot residual analysis
rmse = np.sqrt(np.mean((r_t-est_r_t)**2))

plt.figure()
plt.title('20ms Linear Kernel - Residuals')
resid = r_t-est_r_t
plt.figure()
plt.suptitle('Residual Analysis')
plt.subplot(2,1,1)
plt.plot(resid)
plt.xlabel('Time (ms)')
plt.ylabel('Residual')
plt.subplot(2,1,2)
plt.hist(resid[200:-200], bins=50)
plt.xlabel('Residual')
plt.ylabel('Count')