# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:27:36 2022

@author: Benjamin
"""

#%% Assignment 5 - Q1
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(123)

left = pd.read_csv('leftRate.csv')
left = pd.DataFrame.to_numpy(left).squeeze()
right = pd.read_csv('rightRate.csv')
right = pd.DataFrame.to_numpy(right).squeeze()

plt.figure()
plt.title('Neuron response to left vs right stimuli')
plt.hist(left, bins=20, alpha=0.8, label='Left')
plt.hist(right, bins=20, alpha=0.7, label='Right')
plt.xlabel('response')
plt.ylabel('count')
plt.legend()


fp = np.zeros(83)
tp = np.zeros(83)
for z in np.arange(0,83):
    fp[z] = np.mean(left > z)
    tp[z] = np.mean(right > z)

plt.figure()
plt.title('ROC curve')    
plt.plot(fp,tp)
plt.plot(np.arange(0,1.1), np.arange(0,1.1), color='grey', ls='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

auc = 0
for i in range(82):
    auc += (fp[i]-fp[i + 1])*(np.mean(tp[i:i+1]))
    

