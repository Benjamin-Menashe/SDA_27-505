# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:22:45 2022

@author: Benjamin
"""

# Assignment 2 - Q1
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_pickle('q2data.pkl')

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
    mean_rt = np.mean(data[:,:,200:600], (1,2)) - np.mean(data[:,:,:200], (1,2))
    return mean_rt*1000

