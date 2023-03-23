# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:42:29 2022

@author: Benjamin
"""

# Assignment 2 - Q3
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def StochPointProperties(signal, window_size):
    # Fano Factor: compute spike rate per bin of size window_size (in seconds)
    binned_vec = []
    for i in np.arange(0, np.ceil(signal[-1]/window_size)):
        binned_vec.append(np.count_nonzero(signal >= i*window_size) - np.count_nonzero(signal > (i+1)*window_size))
    
    FF_mean = np.mean(binned_vec)
    FF_var = np.var(binned_vec)
    Fano = FF_var/FF_mean
    
    # Coefficient of variation: compute interval between times
    intervals = np.diff(signal)
    CV_mean = np.mean(intervals)
    CV_std = np.std(intervals)
    CV = CV_std/CV_mean
    
    return Fano, CV
