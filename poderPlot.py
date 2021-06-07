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
 
data_1 = np.random.normal(0.2, 0.1, 10)
data_2 = np.random.normal(0.32, 0.12, 10)
data_3 = np.random.normal(0.31, 0.15, 10)
data_4 = np.random.normal(0.6, 0.24, 10)
data = [data_1, data_2, data_3, data_4]
 
fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
ticks = plt.xticks([1,2,3,4],[0,30,60,100])
plt.title(label="MTSI Positional Verification", fontsize=20)
plt.xlabel("Distance from Isocentre (mm)")
plt.ylabel("Deviation from planned position (mm)")

# show plot
plt.show()