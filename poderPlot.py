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

data_4 = [0.80,	0.20, 0.00,	0.90, 1.20,	0.20, 0.10,	0.70, 0.40,	0.10, 0.80,	0.70]
data_2 = [0.50, 0.20, 0.00, 0.70, 1.10, 0.10, 0.20,	0.10, 0.30,	0.20, 0.50,	0.30]
data_3 = [0.20,	0.40, 0.20,	0.50, 1.20,	0.10, 0.20,	0.40, 0.60,	0.30, 0.70,	0.30]
data_1 = [0.00,	0.20, 0.10,	0.20, 0.70,	0.20, 0.10,	0.30, 0.30,	0.70, 0.20,	0.00]

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
ticks = plt.xticks([1,2,3,4],[30,50,60,100])
plt.title(label="MTSI Positional Verification", fontsize=25)
plt.xlabel("Distance from Isocentre (mm)", fontsize=10)
plt.ylabel("Deviation from DRR (mm)", fontsize=10)

# show plot
plt.show()


