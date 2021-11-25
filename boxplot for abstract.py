# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:16:06 2021

@author: 56153805
"""
import matplotlib.pyplot as plt



#results put in abstract
data_30  = [0.00, 0.20, 0.10, 0.20, 0.70, 0.20, 0.10, 0.30, 0.30, 0.70, 0.20, 0.00]
data_50  = [0.50, 0.20, 0.00, 0.70, 1.10, 0.10, 0.20, 0.10, 0.30, 0.20, 0.50, 0.30]
data_60  = [0.20, 0.40, 0.20, 0.50, 1.20, 0.10, 0.20, 0.40, 0.60, 0.30, 0.70, 0.30]
data_100 = [0.80, 0.20, 0.00, 0.90, 1.20, 0.20, 0.10, 0.70, 0.40, 0.10, 0.80, 0.70]

# Creating plot

# in order of distance from isocentre
data = [data_30, data_50, data_60, data_100]
    
fig = plt.figure(figsize =(7, 4))
ax = fig.add_axes([0, 0, 1, 1])

csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}
    
bp = ax.boxplot(data)
ticks = plt.xticks([1,2,3,4],[0,50,60,100], fontsize = 15)
plt.yticks(fontsize = 15)
plt.title(label="MTSI Positional Verification", fontsize=25)
plt.xlabel("Distance from Isocentre (mm)", fontsize=20)
plt.ylabel("Deviation from Target (mm)", fontsize=20)
