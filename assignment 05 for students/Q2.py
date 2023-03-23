# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:05:03 2022

@author: Benjamin
"""
#%% Assignment 5 - Q2
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(123)

#%% A (added defaults just for testing)
def roachNeurons(r0, n, ai, a):
# The inputs of the code:
#     𝑟0 – baseline firing rate
#     n – number of interneurons
#     ai – vector of the preferred angle 𝛼 of each neuron
#     a – real angle
# The output of the code:
#     a_hat – the prediction of the neurons using population vector
    neurons = [np.maximum(r0*np.cos(a-ai[i]),0) for i in range(n)]
    x_hat = np.sum(neurons*np.sin(ai)/r0)
    y_hat = np.sum(neurons*np.cos(ai)/r0)
    a_hat = np.arctan2(x_hat, y_hat) % (2*np.pi)
    
    return a_hat, neurons


#%% B - 1
r0 = 55
n=3
ai = np.asarray([0, 2*np.pi/3, 4*np.pi/3])
a_hat = np.zeros(int(1000*2*np.pi))
neurons = np.zeros((int(1000*2*np.pi),n))

for i in range(int(1000*2*np.pi)):
    a_hat[i], neurons[i] = roachNeurons(r0, n, ai, i/1000)
    
plt.figure()
plt.title('Predicted Angle - 3 interneurons')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(a_hat))
plt.xlabel('actual angle (degrees)')
plt.ylabel('predicted angle (degrees)')

plt.figure()
plt.title('Interneuron Activity - 3 interneurons')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),neurons[:,0], label='neuron 1')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),neurons[:,1], label='neuron 2')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),neurons[:,2], label='neuron 3')
plt.xlabel('actual angle (degrees)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()

#%% B - 2
plt.figure()
for i in range(4,11):
    error = np.zeros(int(1000*2*np.pi))
    for j in range(int(1000*2*np.pi)):
        a_hat,_ = roachNeurons(55, i, np.arange(0,1,1/i)*2*np.pi, j/1000)
        error[j] = j/1000 - a_hat
    
    plt.plot(np.degrees(np.arange(0.001,2*np.pi-0.001,0.001)), error[1:], label=f"{i}")

plt.title('Angle Prediction Errors for Different Numbers of Interneurons')
plt.xlabel('actual angle (degrees)')
plt.ylabel('error (degrees)')
plt.legend()

#%% C

def roachNeurons_noise(r0, n, ai, a):
# The inputs of the code:
#     𝑟0 – baseline firing rate
#     n – number of interneurons
#     ai – vector of the preferred angle 𝛼 of each neuron
#     a – real angle
# The output of the code:
#     a_hat – the prediction of the neurons using population vector
    neurons = [np.maximum(r0*np.cos(a-ai[i])+np.random.normal(0,4),0) for i in range(n)]
    x_hat = np.sum(neurons*np.sin(ai)/r0)
    y_hat = np.sum(neurons*np.cos(ai)/r0)
    a_hat = np.arctan2(x_hat, y_hat)
    
    if a > 0.50 and a < 5.5:
        a_hat = a_hat % (2*np.pi)
    elif a >= 5.5:
        a_hat += 2*np.pi
 
    return a_hat

# try with 4 neurons
r0 = 55
n = 4
ai = np.arange(0,1,1/n)*2*np.pi
a_hat = np.zeros((100,int(1000*2*np.pi)))
rmse = np.zeros(int(1000*2*np.pi))
for i in range(100):
    for j in range(int(1000*2*np.pi)):
        a_hat[i,j] = roachNeurons_noise(r0, n, ai, j/1000)
        rmse[j] += np.sqrt((a_hat[i,j]-j/1000)**2)

mean_a_hat = np.mean(a_hat,0)
std_a_hat = np.std(a_hat,0)
rmse = rmse/100

plt.figure()
plt.title('Predicted Angle Under Noisy Neurons - 4 neurons')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat))
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat + 2.5*std_a_hat), color='grey', lw=0.5, ls='--')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat - 2.5*std_a_hat), color='grey', lw=0.5, ls='--')
plt.xlabel('actual angle (degrees)')
plt.ylabel('predicted angle (degrees)')
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)), rmse)
plt.title('RMSE per Angle - 4 neurons')
plt.xlabel('actual angle (degrees)')
plt.ylabel('mean RMSE')

# try with 7 neurons
r0 = 55
n = 7
ai = np.arange(0,1,1/n)*2*np.pi
a_hat = np.zeros((100,int(1000*2*np.pi)))
rmse = np.zeros(int(1000*2*np.pi))
for i in range(100):
    for j in range(int(1000*2*np.pi)):
        a_hat[i,j] = roachNeurons_noise(r0, n, ai, j/1000)
        rmse[j] += np.sqrt((a_hat[i,j]-j/1000)**2)

mean_a_hat = np.mean(a_hat,0)
std_a_hat = np.std(a_hat,0)
rmse = rmse/100

plt.figure()
plt.title('Predicted Angle Under Noisy Neurons - 7 neurons')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat))
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat + 2.5*std_a_hat), color='grey', lw=0.5, ls='--')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat - 2.5*std_a_hat), color='grey', lw=0.5, ls='--')
plt.xlabel('actual angle (degrees)')
plt.ylabel('predicted angle (degrees)')
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)), rmse)
plt.title('RMSE per Angle - 7 neurons')
plt.xlabel('actual angle (degrees)')
plt.ylabel('mean RMSE')


# try with 12 neurons
r0 = 55
n = 12
ai = np.arange(0,1,1/n)*2*np.pi
a_hat = np.zeros((100,int(1000*2*np.pi)))
rmse = np.zeros(int(1000*2*np.pi))
for i in range(100):
    for j in range(int(1000*2*np.pi)):
        a_hat[i,j] = roachNeurons_noise(r0, n, ai, j/1000)
        rmse[j] += np.sqrt((a_hat[i,j]-j/1000)**2)

mean_a_hat = np.mean(a_hat,0)
std_a_hat = np.std(a_hat,0)
rmse = rmse/100

plt.figure()
plt.title('Predicted Angle Under Noisy Neurons - 12 neurons')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat))
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat + 2.5*std_a_hat), color='grey', lw=0.5, ls='--')
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)),np.degrees(mean_a_hat - 2.5*std_a_hat), color='grey', lw=0.5, ls='--')
plt.xlabel('actual angle (degrees)')
plt.ylabel('predicted angle (degrees)')
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.degrees(np.arange(0,2*np.pi-0.001,0.001)), rmse)
plt.title('RMSE per Angle - 12 neurons')
plt.xlabel('actual angle (degrees)')
plt.ylabel('mean RMSE')