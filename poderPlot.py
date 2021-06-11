# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:20:28 2021

@author: 56153805
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# Creating dataset
np.random.seed(10)

data_1 = [0.0, 0.4, 0.5, 0.6]
data_2 = [0.4, 0.6, 0.9, 0.5]
data_3 = [0.7, 0.6, 0.4, 1.0]
data_4 = [0.7, 1.2, 0.5, 1.5]

# data_1 = np.random.normal(0.2, 0.1, 10)
# data_2 = np.random.normal(0.32, 0.12, 10)
# data_3 = np.random.normal(0.31, 0.15, 10)
# data_4 = np.random.normal(0.6, 0.24, 10)
data = [data_1, data_2, data_3, data_4]
 
fig = plt.figure(figsize =(7, 5))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
ticks = plt.xticks([1,2,3,4],[0,30,60,100])
plt.title(label="MTSI Positional Verification", fontsize=25)
plt.xlabel("Distance from Isocentre (mm)", fontsize=25)
plt.ylabel("Deviation from DRR (mm)", fontsize=25)

# show plot
plt.show()