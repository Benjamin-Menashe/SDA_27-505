# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:55:03 2022

@author: Benjamin
"""
# Assignment 1 - Question 3c
import numpy as np

def MyConv(signal, window):
    pad = np.zeros(len(window)-1)
    full_signal = np.concatenate((pad, signal, pad))
    
    output = np.zeros(len(full_signal)-len(pad))
    for i in np.arange(len(full_signal)-len(pad)):
        for j in np.arange(len(window)): 
            output[i] += full_signal[i+j]*window[len(window)-j-1]
            
    return output
        
    