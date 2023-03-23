import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

n = 1000000
p = 50/1000
t = np.arange(-500,500+1)

# Poisson-like, 50 Hz, C reduces A and B to 5 hz during 4ms-24ms after spike
C = np.zeros(n) # spike train C
for i in range(n):
    C[i] = np.random.binomial(1,p)

A = np.zeros(n) # spike train A
for i in range(24, n):
    p = 5/1000 if np.any(C[i-24:i-4]) else 50/1000
    A[i] = np.random.binomial(1,p)
    
B = np.zeros(n) # spike train B
for i in range(24, n):
    p = 5/1000 if np.any(C[i-24:i-4]) else 50/1000
    B[i] = np.random.binomial(1,p)
    
AB_cc = np.zeros(1000+1) # CC(A,B)
for i in range(-500,500):
    AB_cc[500+i] = np.sum(np.logical_and(A[500:-500], B[i+500:-500+i]))/np.sum(A[500:-500])    
AB_cc = AB_cc*1000
AB_cc[-1] =  AB_cc[-2]

plt.figure()
plt.title('Cross-correlation (A, B)')
plt.plot(t, AB_cc)
plt.xlabel('time after spike (ms)')
plt.ylabel('Firing Rate (hz)')
plt.show()

AC_cc = np.zeros(1000+1) # CC(A,C)
for i in range(-500,500):
    AC_cc[500+i] = np.sum(np.logical_and(A[500:-500], C[i+500:-500+i]))/np.sum(A[500:-500])    
AC_cc = AC_cc*1000
AC_cc[-1] =  AC_cc[-2]

plt.figure()
plt.title('Cross-correlation (A, C)')
plt.plot(t, AC_cc)
plt.xlabel('time after spike (ms)')
plt.ylabel('Firing Rate (hz)')
plt.show()

BC_cc = np.zeros(1000+1) # CC(B,C)
for i in range(-500,500):
    BC_cc[500+i] = np.sum(np.logical_and(B[500:-500], C[i+500:-500+i]))/np.sum(A[500:-500])    
BC_cc = BC_cc*1000
BC_cc[-1] =  BC_cc[-2]

plt.figure()
plt.title('Cross-correlation (B, C)')
plt.plot(t, BC_cc)
plt.xlabel('time after spike (ms)')
plt.ylabel('Firing Rate (hz)')
plt.show()


