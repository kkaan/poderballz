# -*- coding: utf-8 -*-
"""
Margin calculation for IGRT matches

Created on Mon Nov 22 09:25:14 2021

@author: 56153805
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as st



filename = 'P:/09 Staff/Kaan/CBCT image matching.xlsx'
df_matchdata = pd.read_excel(filename, header = 1)

#convert translations to mm 
df_matchdata.iloc[:,[1,2,3]] = df_matchdata.iloc[:,[1,2,3]]*10


maximum_OAD = 150 #mm
interval = 10 #mm
n_fx = df_matchdata.shape[0]

oads = np.array(range(0, maximum_OAD+interval, interval))
n_targets = oads.size       
oads_dev = np.zeros((n_fx, n_targets))
oads_xyz = np.zeros((oads.size, 4)) #sum of squared x y z deviations for each QAD
oads_xyz[:,0] = oads


df_projected_dev = pd.DataFrame(oads_dev, columns = oads)




for oad_idx, oad in enumerate(oads):
    
    for fraction, row in df_matchdata.iterrows():
    
        pitch = row.loc['PIT']   # pitch angle
        roll = row.loc['ROLL']  # roll angle
        rtn = row.loc['RTN'] # yaw angle
        
        vert = row.loc['VRT']
        long = row.loc['LNG']
        lat = row.loc['LAT']
        
        pitch = np.radians(pitch)
        roll = np.radians(roll)
        rtn = np.radians(rtn)
        
        R_pitch = np.array(( (1, 0, 0, 0),
                             (0, np.cos(pitch), -np.sin(pitch), 0),
                             (0, np.sin(pitch), np.cos(pitch), 0),
                             (0, 0, 0, 1) ))
        
        R_roll  = np.array(( (np.cos(roll), 0, np.sin(roll), 0),
                             (0, 1, 0, 0),
                             (-np.sin(roll), 0, np.cos(roll), 0),
                             (0, 0, 0, 1) ))
        
        R_rtn  =  np.array(( (np.cos(rtn), -np.sin(rtn), 0, 0),
                             (np.sin(rtn), np.cos(rtn), 0, 0),
                             (0, 0, 1, 0),
                             (0, 0, 0, 1) ))
        
        # translation offsets in mm
        
        
        T =       np.array(( (1, 0, 0, lat),
                             (0, 1, 0, long),
                             (0, 0, 1, vert),
                             (0, 0, 0, 1) ))
        
        
        # rx, ry and rz are distance between iso and target centre
        
        rx = ry = rz = np.sqrt(np.power(oad,2)/3)
            
        r = np.array((rx, ry, rz, 1))
        
        r_new = R_pitch @ R_roll @ R_rtn @ T @ r
        
        E = r_new-r
        oads_xyz[oad_idx,1:4] =  oads_xyz[oad_idx,1:4] + np.square(E[0:3])
        delta = np.sqrt(E.dot(E))
        oads_dev[fraction, oad_idx] = delta

sigma_mean0 = np.sqrt(oads_xyz[:,1:4]/n_fx)
ortega_margin = sigma_mean0*2.5

# ax = df_projected_dev.boxplot(showfliers=False)
# ax.set_ylabel('Deviation from planned position (mm)')
# ax.set_xlabel('Distance of target from isocentre (mm)')
# for i in oads:
#     y = df_projected_dev.loc[:,i]
#     x = np.random.normal(1+i, 0.04, size=len(y))
#     ax.plot(x, y, 'r.', alpha=0.2)

# Set a flag to display a particular experiment:
# df_matchdata["DISPLAY"]= np.zeros(len(df_matchdata))
# df_matchdata.loc[4,"DISPLAY"] = 1

indexlist = [0]* len(df_matchdata)
highlightsample = 4
indexlist[highlightsample] = 1
df_matchdata.index = indexlist
df_matchdata.index.name = "Index"



## Ortega
# n_j = df_projected_dev.shape[0] #number of experiments

# sigma_mean0 = (df_projected_dev.pow(2).sum() / n_j).pow(0.5)

# ortega_margin = sigma_mean0*2.5


# Plot projected deviation
sns.set(font_scale=3, rc={'figure.figsize':(30,15)})
sns.set_style("whitegrid")
ax = sns.boxplot(data=(df_projected_dev), 
                      showfliers=False, palette="flare", whis=[0,90])
np.random.seed(123)
ax = sns.stripplot(data=df_projected_dev, marker="o", 
                   alpha=0.3, color="black",size=6)

ax = sns.lineplot(x=oads.astype(str), y=ortega_margin[:,0], linewidth = 10, label="Ortega_x")
ax = sns.lineplot(x=oads.astype(str), y=ortega_margin[:,1], linewidth = 10, label="Ortega_y")
ax = sns.lineplot(x=oads.astype(str), y=ortega_margin[:,2], linewidth = 10, label="Ortega_x")

ax.set_xlabel("Distance from isocentre (mm)", labelpad=20)
ax.set_ylabel("Deviation (mm)", labelpad=20)

plt.ylim(None, 5)
plt.legend(loc='upper left')



#plt.xlim(None, 12.5)
plt.show()




# Boxplot the match results for each dimension
 
# Create an array with the colors you want to use
colors = ["#000000", "#FF0000"]

sns.set(font_scale=3, rc={'figure.figsize':(30,15)})
sns.set_style("whitegrid")
ax = sns
ax = sns.boxplot(data=df_matchdata.drop("TREATMENT DATE", axis=1), 
                palette="viridis")
dfm = df_matchdata.drop("TREATMENT DATE", axis=1).reset_index().melt('Index')
np.random.seed(123) #this keep the jitted the same when plotting
ax = sns.stripplot(data=dfm, x="variable", y="value", color="black", jitter = True, 
                   marker="o", alpha=0.3, size=13)
# ax = ax.map(sns.stripplot(data=df_matchdata.loc[4,:], marker="o", alpha=1, 
#                         color="red",size=13))
ax.set_xlabel("Axis of translation/rotation", labelpad=20)
ax.set_ylabel("Deviation (mm or degrees)", labelpad=20)
plt.show()


sns.set(font_scale=3, rc={'figure.figsize':(30,15)})
sns.set_style("whitegrid")
ax = sns
ax = sns.boxplot(data=df_matchdata.drop("TREATMENT DATE", axis=1), 
                 showfliers=False,  palette="viridis")
dfm = df_matchdata.drop("TREATMENT DATE", axis=1).reset_index().melt('Index')
np.random.seed(123) #this keep the jitted the same when plotting
ax = sns.stripplot(data=dfm, x="variable", y="value", hue="Index", palette=colors, jitter = True, 
                   marker="o", alpha=0., size=13)
# ax = ax.map(sns.stripplot(data=df_matchdata.loc[4,:], marker="o", alpha=1, 
#                         color="red",size=13))
ax.set_xlabel("Axis of translation/rotation", labelpad=20)
ax.set_ylabel("Deviation (mm or degrees)", labelpad=20)
plt.show()



