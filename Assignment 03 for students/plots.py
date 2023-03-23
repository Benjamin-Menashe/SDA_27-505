# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:39:17 2022

@author: Benjamin
"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.title('Autocorrelation')
plt.hlines(0, -1000, 1000, lw = 0)
plt.xlabel('time after spike (ms)')
ax = plt.gca()
ax.axes.yaxis.set_ticks([])
plt.show()

plt.figure()
plt.title('Autocorrelation - Neuron A')
plt.hlines(0, -500, 500, lw = 0)
plt.xlabel('time after spike (ms)')
plt.ylim((0, 40))
plt.ylabel('Firing Rate (hz)')
plt.show()

plt.figure()
plt.title('Autocorrelation - Neuron B')
plt.hlines(0, -500, 500, lw = 0)
plt.xlabel('time after spike (ms)')
plt.ylim((0, 40))
plt.ylabel('Firing Rate (hz)')
plt.show()

plt.figure()
plt.title('Cross-correlation (A, B)')
plt.hlines(0, -500, 500, lw = 0)
plt.xlabel('time after spike (ms)')
plt.ylim((0, 40))
plt.ylabel('Firing Rate (hz)')
plt.show()