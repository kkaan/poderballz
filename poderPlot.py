# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:20:28 2021

@author: 56153805
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# # Creating dataset
# np.random.seed(10)

# static HTT results from commissioning:

# data_4 = [0.80,	0.20, 0.00,	0.90, 1.20,	0.20, 0.10,	0.70, 0.40,	0.10, 0.80,	0.70]
# data_2 = [0.50, 0.20, 0.00, 0.70, 1.10, 0.10, 0.20,	0.10, 0.30,	0.20, 0.50,	0.30]
# data_3 = [0.20,	0.40, 0.20,	0.50, 1.20,	0.10, 0.20,	0.40, 0.60,	0.30, 0.70,	0.30]
# data_1 = [0.00,	0.20, 0.10,	0.20, 0.70,	0.20, 0.10,	0.30, 0.30,	0.70, 0.20,	0.00]


# data_1 = np.random.normal(0.2, 0.1, 10)
# data_2 = np.random.normal(0.32, 0.12, 10)
# data_3 = np.random.normal(0.31, 0.15, 10)
# data_4 = np.random.normal(0.6, 0.24, 10)


data_1 = (df['WL', 1,'x']**2*df['WL', 1,'y']**2)**(1/2)
data_2 = (df['WL', 2,'x']**2*df['WL', 2,'y']**2)**(1/2)
data_3 = (df['WL', 3,'x']**2*df['WL', 3,'y']**2)**(1/2)
data_4 = (df['WL', 4,'x']**2*df['WL', 4,'y']**2)**(1/2) 

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



## removing outliers

what = 'EPIDBalls'


window = 5



x = df[what, 1, 'x'].copy()

# add values to front for rolling average to work at start.
m = np.pad(x, pad_width=(window-1, windoow-1), mode='wrap')
m = pandas.Series(m)
m = m.rolling(window).median()

# crop the NAN values
m = m.iloc[window-1:]
m.reset_index(drop=True, inplace=True)

s = np.pad(x, pad_width=(window-1, 0), mode='wrap')
s = pandas.Series(s)
s = s.rolling(window).std()
s = s.iloc[window-1:]
s.reset_index(drop=True, inplace=True)


x = x.mask((x <= m+2*s) & (x >=m-2*s))

x2 = x[(x <= m+2*s) & (x >=m-2*s)]

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('window = {}'.format(window))
x.plot(ax=ax1)
x2.plot(ax=ax2)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('PoderBallz A tale of four balls {}'.format(what))
colours = ['darkred', 'darkblue', 'darkgreen', 'darkslategrey']
# need more colours for more balls.

# num_of_balls = 1 # comment out this line to print all balls

for i in range(1,num_of_balls+1):
    df.plot(kind="scatter", x='Gantry', y=(what, i, 'x'), s=1,
            color=colours[i-1], label='ball {}'.format(i), ax=ax1)
    df.plot(kind="scatter", x='Gantry', y='xsmoothed', s=1,
            color=colours[i], label='ball {}'.format(i), ax=ax1)
    df.plot(kind="scatter", x='Gantry', y=(what, i, 'y'), s=1,
            color=colours[i-1], label='ball {}'.format(i), ax=ax2)

ax1.set_ylabel('X position (mm)')  
ax2.set_ylabel('Y position (mm)')

ax1.legend(title="None", fontsize="xx-small", loc="upper right")
ax2.get_legend().remove()

plt.show()