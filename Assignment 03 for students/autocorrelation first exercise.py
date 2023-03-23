# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:39:57 2022

@author: yalud
"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as tsa

spikeTrain=np.zeros(650)
firstSpikesIdices=np.linspace(0,600,num=7).astype(int).tolist()
secondSpikesIdices=np.linspace(5,605,num=7).astype(int).tolist()
np.put(spikeTrain,firstSpikesIdices,np.ones(7))
np.put(spikeTrain,secondSpikesIdices,np.ones(7))

plt.figure(1)
plt.plot(range(650),spikeTrain)
plt.ylabel('spike (AU)')
plt.xlabel('Time (ms)')
plt.title('Spike train')


CDF=np.zeros(100)
np.put(CDF,range(5,95),np.ones(90)*0.5)
np.put(CDF,range(95,100),np.ones(6)*1)
plt.figure(2)
plt.plot(range(100),CDF)
plt.ylabel('probability')
plt.xlabel('Time (ms)')
plt.title('CDF')

hazard=np.zeros(100)
np.put(hazard,5,0.5)
np.put(hazard,95,1)
plt.figure(3)
plt.plot(range(100),hazard)
plt.ylabel('probability')
plt.xlabel('Time (ms)')
plt.title('Hazard')

autocorrelation=tsa.acf(spikeTrain,nlags=250)
autocorrelation[0]=0
plt.figure(4)
plt.plot(range(251), autocorrelation)
plt.yticks(np.linspace(0,1,num=11))
plt.ylabel('probability')
plt.xlabel('Time lag (ms)')
plt.title('Autocorrelation')