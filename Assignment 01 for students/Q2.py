# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:59:28 2022

@author: Benjamin
"""


# SDA_2022 Assignment 01
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
np.random.seed(123)

#%% Q3a

spk_trn = np.zeros(10000)
# import spk_time manually
spk_tms = np.floor(1000*spk_timescsv).astype(int)
spk_trn[spk_tms] = 1

mean_all = sum(spk_trn)/10

#%% Q3b
t = np.arange(10000)

mean_200 = np.zeros(10000)
for i in np.arange(0,10000,200):
    mean_200[i:i+200] = sum(spk_trn[i:i+200]) / (0.2)

mean_500 = np.zeros(10000)
for i in np.arange(0,10000,500):
    mean_500[i:i+500] = sum(spk_trn[i:i+500]) / (0.5)

mean_1000 = np.zeros(10000)
for i in np.arange(0,10000,1000):
    mean_1000[i:i+1000] = sum(spk_trn[i:i+1000]) / (1)

mean_3000 = np.zeros(10000)
for i in np.arange(0,10000,3000):
    mean_3000[i:i+3000] = sum(spk_trn[i:i+3000]) / (3)

plt.figure
plt.title('Mean Firing Rate for Different Windows')
plt.plot(t, mean_200, label = "0.2 s")
plt.plot(t, mean_500, label = "0.5 s")
plt.plot(t, mean_1000, label = "1 s")
plt.plot(t, mean_3000, label = "3 s")
plt.ylabel('firing Rate per second')
plt.xlabel('time (s)')
plt.legend()

#%% Q3c
def MyConv(signal, window):
    pad = np.zeros(len(window)-1)
    full_signal = np.concatenate((pad, signal, pad))
    
    output = np.zeros(len(full_signal)-len(pad))
    for i in np.arange(len(full_signal)-len(pad)):
        for j in np.arange(len(window)): 
            output[i] += full_signal[i+j]*window[len(window)-j-1]
            
    return output

signal = np.ones(30)
window = np.ones(6)/6

output = MyConv(signal, window)
output_auto = np.convolve(signal, window)

plt.figure
plt.subplot(2,1,1)
plt.title('MyConv', fontsize = 10)
plt.plot(output)
plt.tight_layout(pad = 2)
plt.subplot(2,1,2)
plt.title('numpy.convolve', fontsize = 10)
plt.plot(output_auto)
#%% Q3d

window1 = np.ones(200)/200
output1 = MyConv(spk_trn, window1)*1000

window2 = np.ones(500)/500
output2 = MyConv(spk_trn, window2)*1000

window3 = np.ones(1000)/1000
output3 = MyConv(spk_trn, window3)*1000

window4 = np.ones(3000)/3000
output4 = MyConv(spk_trn, window4)*1000


plt.figure
plt.title('Mean Firing Rate for Sliding Rectangular Windows')
plt.plot(t, output1[99:-100], label = "0.2 s")
plt.plot(t, output2[249:-250], label = "0.5 s")
plt.plot(t, output3[499:-500], label = "1 s")
plt.plot(t, output4[1499:-1500], label = "3 s")
plt.ylabel('firing Rate per second')
plt.xlabel('time (s)')
plt.legend()


#%% Q3e

gwindow1 = sig.windows.gaussian(800, 250)
goutput1 = MyConv(spk_trn, gwindow1)*1000/sum(gwindow1)

gwindow2 = sig.windows.gaussian(450, 100)
goutput2 = MyConv(spk_trn, gwindow2)*1000/sum(gwindow2)

plt.figure
plt.title('Mean Firing Rate for Sliding Gaussian Windows')
plt.plot(t, goutput1[399:-400], label = "span = 0.80s, std = 0.25s")
plt.plot(t, goutput2[224:-225], label = "span = 0.45s, std = 0.10s")
plt.ylabel('firing Rate per second')
plt.xlabel('time (s)')
plt.legend()

plt.figure
plt.title('Gaussian Windows Used')
plt.plot(gwindow1/sum(gwindow1), label = "span = 0.80s, std = 0.25s")
plt.plot(gwindow2/sum(gwindow2), label = "span = 0.45s, std = 0.10s")
plt.legend()


#%%

plt.eventplot(spk_timescsv, linewidths=0.2)
plt.title('10 s Spike Train')
ax = plt.gca()
ax.get_yaxis().set_visible(False)
