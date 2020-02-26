# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:05:39 2019

@author: nipun
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## Log Likelihood function
def log_likelihood(x, y, b0, b1):
    result = -(np.log(1 + np.exp(-y*(b0 + b1*x))))
    return result


b0 = np.arange(-2.0, 2.1, 0.1)
b1 = np.arange(-2.0, 2.1, 0.1)
ll_1 = np.zeros((41,41))
ll_2 = np.zeros((41,41))


x = 1

B0, B1 = np.meshgrid(b0, b1)

## Computation of Log-Likelihood
for i in range(len(b0)):
    for j in range(len(b0)):
        ll_1[i][j] = log_likelihood(x, -1, B0[i][j], B1[i][j])
        ll_2[i][j] = log_likelihood(x, 1, B0[i][j], B1[i][j])
        

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot_wireframe(B0, B1, ll_1)

ax.set_xlabel('B0')
ax.set_ylabel('B1')
ax.set_zlabel('Log-likelihood')
ax.set_title('Log-likelihood when Y = -1')




fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(B0, B1, ll_2)

ax.set_xlabel('B0')
ax.set_ylabel('B1')
ax.set_zlabel('Log-likelihood')
ax.set_title('Log-likelihood when Y = 1')
