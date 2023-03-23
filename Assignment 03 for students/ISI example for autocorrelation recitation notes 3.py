# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 09:32:54 2022

@author: yalud
"""

isiVec=np.array([3,2,1,3,1,2,3])
weights = np.ones_like(isiVec)/float(len(isiVec))
plt.figure(1)
plt.hist(isiVec,weights=weights)
plt.xticks(range(5))
plt.ylabel('probability')
plt.xlabel('ISI')
plt.title('TIH')
