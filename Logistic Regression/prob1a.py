# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:05:39 2019

@author: nipun
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## Sigmoid Function
def sigmoid_func(X):
    sigmoid = 1/(1+np.exp(-X))
    return sigmoid


b0 = np.arange(-2.0, 2.1, 0.1)
b1 = np.arange(-2.0, 2.1, 0.1)
x = 1
pb = np.zeros((41,41))

B0, B1 = np.meshgrid(b0, b1)


for i in range(len(b0)):
    for j in range(len(b0)):
        pb[i][j] = sigmoid_func(B0[i][j] + B1[i][j]*x)
    
fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot_wireframe(B0, B1, pb)

ax.set_xlabel('B0')
ax.set_ylabel('B1')
ax.set_zlabel('Probability')
ax.set_title('3D Plot of Probability with respect to B0 and B1')