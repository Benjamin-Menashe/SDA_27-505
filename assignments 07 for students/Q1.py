# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 08:38:38 2022

@author: Benjamin
"""

# SDA_2022 Assignment 07 Q1
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
np.random.seed(123)

#%% 1C
def best_bin_calculator(spk_mat):

    n = np.shape(spk_mat)[0]
    t = np.shape(spk_mat)[1]
    h_b_max = 0
    best_bin = 0
    
    for i in range(5,100):
        sums = spk_mat[:,:i*int(np.floor(t/i))].reshape((n,int(np.floor(t/i)),i)).sum(axis=2)
        h_t = sp.entropy(np.bincount(sums.flatten().astype(int))/i)
        noise = np.mean([sp.entropy(np.bincount(sums[:,j].astype(int))/n) for j in range(int(np.floor(t/i)))])
        h_b = h_t - noise
        if h_b_max < h_b:
            h_b_max = h_b
            best_bin = i
            
    return best_bin, h_b_max

#%% 1D
spk_trns = np.zeros((3,2000))
fr_1 = 0.1 # change here
fr_2 = 0.0 # change here
s = np.zeros(100)

for k in range(100):

    for i in range(2000):
        if (i % 100) < 50:
            spk_trns[:,i] = np.random.binomial(1, fr_1, size=3)
        else:
            spk_trns[:,i] = np.random.binomial(1, fr_2, size=3)
    
    s[k] = best_bin_calculator(spk_trns)[0]

print(np.sum(s==50)/100, np.mean(s), np.std(s))

#%% 1Ea

spk_mat = np.load('assignments 07 for students/spk_mat.npy')
best_bin = best_bin_calculator(spk_mat)[0]
print(best_bin)

#%% 1Eb
entropies = np.zeros((8,2))
for i in range(8):
    neuron = spk_mat[i,:].reshape(8,9000)
    entropies[i,:] = best_bin_calculator(neuron)

print(np.round(entropies,2))

#%% 1Ec - 1
# mutual information function

def Mutual_Information_2_Neurons(spk_mat, n1, n2, bin_size = 100):
    bs = bin_size
    t = np.shape(spk_mat)[1]
    joint_dist = np.zeros((bs,bs))
    marg1 = np.zeros(bs)
    marg2 = np.zeros(bs)
    
    sums = spk_mat[(n1,n2),:bs*int(np.floor(t/bs))].reshape((2,int(np.floor(t/bs)),bs)).sum(axis=2)
    for i in range(int(np.floor(t/bs))):
        joint_dist[sums[0,i],sums[1,i]] += 1
        marg1[sums[0,i]] += 1
        marg2[sums[1,i]] += 1
    joint_dist = joint_dist/(np.floor(t/bs))
    marg1 = marg1/(np.floor(t/bs))
    marg2 = marg2/(np.floor(t/bs))
    
    uncond_dist = np.outer(marg1, marg2)
    
    mutual_I = sp.entropy(joint_dist.flatten(), uncond_dist.flatten())
    
    plt.figure()
    plt.imshow(joint_dist, origin='lower')
    plt.title(f"Joint Distribution neurons {n1+1} and {n2+1}")
    plt.xlabel(f"Neuron {n1+1} Firing Rate per bin")
    plt.ylabel(f"Neuron {n2+1} Firing Rate per bin")
    
    return mutual_I

#%% 1Ec - 2
# for neuron 3

n3 = np.zeros(8)
for i in range(8):
    n3[i] = Mutual_Information_2_Neurons(spk_mat, 2, i)

n7 = np.zeros(8)
for i in range(8):
    n7[i] = Mutual_Information_2_Neurons(spk_mat, 6, i)

n8 = np.zeros(8)
for i in range(8):
    n8[i] = Mutual_Information_2_Neurons(spk_mat, 7, i)

print(f"neuron 3: {np.argsort(n3)[-2]+1}, neuron 7: {np.argsort(n7)[-2]+1}, neuron 8: {np.argsort(n8)[-2]+1}")