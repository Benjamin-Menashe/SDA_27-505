# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:21:07 2022

@author: Benjamin
"""

# Assignment 2 - Q1
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def SNRseperate(signal, w_size):
    std_vec = [0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500]
    best_SNR = 1
    w_std = 0.5
    for std in std_vec:
        gwindow = sig.windows.gaussian(w_size, std)
        gwindow = gwindow/sum(gwindow)
        data = np.convolve(signal, gwindow, mode='same')
        noise = signal - data
        cur_SNR = 20*np.log10(np.sqrt(np.sum(signal**2))/np.sqrt(np.sum(noise**2)))
        if abs(best_SNR-10) > abs(cur_SNR-10):
            best_SNR = cur_SNR
            w_std = std
        
    gwindow = sig.windows.gaussian(w_size, w_std)
    gwindow = gwindow/sum(gwindow)
    best_data = np.convolve(signal, gwindow, mode='same')
    
    # plot
    plt.figure()
    plt.title(f"Signal before and after cleaning (window size={w_size})")
    plt.plot(signal, label='signal')
    plt.plot(best_data, label = 'clean')
    plt.legend()
    plt.xlabel('time (ms)')
    plt.show()
    
    plt.figure()
    plt.title(f"Residuals (window size={w_size})")
    plt.plot(noise)
    plt.xlabel('time (ms)')
    plt.show()
    
    return best_SNR, w_std, best_data
