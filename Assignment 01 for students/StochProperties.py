# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 22:43:12 2022

@author: Benjamin
"""
# Assignment 1 - Question 2a
import numpy as np

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