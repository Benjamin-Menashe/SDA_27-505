# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:05:03 2022

@author: Benjamin
"""
#%% Assignment 5 - Q2
import numpy as np

#%% A
def roachNeurons(r0, n, ai, a):
# The inputs of the code:
#     𝑟0 – baseline firing rate
#     n – number of interneurons
#     ai – vector of the preferred angle 𝛼 of each neuron
#     a – real angle
# The output of the code:
#     a_hat – the prediction of the neurons using population vector
    neurons = [np.maximum(r0*np.cos(a-ai[i]),0) for i in range(n)] # firing rate per neuron
    x_hat = np.sum(neurons*np.sin(ai)/r0) # decode x direction
    y_hat = np.sum(neurons*np.cos(ai)/r0) # decode y direction
    a_hat = np.arctan2(x_hat, y_hat) % (2*np.pi) # vector to angle
    
    return a_hat


