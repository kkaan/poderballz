# -*- coding: utf-8 -*-
"""
Interpolate the positional data to match the required gantry positions.

Created on Sun Oct  3 16:27:47 2021

@author: 56153805
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from poderPlot import plot_against_gantry

def remove_outliers(df, item_type):
    """
    Remove outliers using a rolling z-values method.

    Parameters
    ----------
    df : Pandas Dataframe
        DESCRIPTION.
    what : String
        DRRBall, DRRApetures, EPIDBalls, EPIDApetures.

    Returns
    -------
    None.

    """
    #what = 'EPIDBalls'
    item_type = ['EPIDBalls', 'EPIDApertures', 'DRRBalls', 'DRRApertures']
    df.loc[:, item_type] = df.loc[:, item_type].astype(float)
    
    # convert pixel values to mm 
    SDD = 500 #mm from iso
    pixel_res = 0.336 #asi1000 = 0.34, asi1200 = 0.39
    pixel_to_mm = pixel_res*(1000+SDD)/1000
    df.loc[:,item_type]= df.loc[:,item_type]*pixel_to_mm
    
    # Use z-score 
    window = 10
    num_stds = 3
    
    
    # pad values to front for rolling average to not result in Nan edge.
    x = np.pad(df.loc[:,item_type], ((window-1, 0),(0,0)), 'wrap')
    x = pandas.DataFrame(x)
    m = x.rolling(window, axis=0).median()
    
    # crop the NaN edges (should be in padded area)
    m = m.iloc[window-1:]
    m.reset_index(drop=True, inplace=True)
    x = x.iloc[window-1:]
    x.reset_index(drop=True, inplace=True)
    
    s = np.pad(m, ((window-1, 0),(0,0)), 'wrap')
    s = pandas.DataFrame(s)
    s = s.rolling(window, axis=0).std()
    
    # crop the NaN opening edge (should only be in padded area)
    s = s.iloc[window-1:]
    s.reset_index(drop=True, inplace=True)
    
    xbool = (x <= m+num_stds*s) & (x >= m-num_stds*s)
    x = x.mask(~xbool)
    
    # convert pixel values to mm 
    SDD = 500 #mm from iso
    pixel_res = 0.336 #asi1000 = 0.34, asi1200 = 0.39
    pixel_to_mm = pixel_res*(1000+SDD)/1000
    df= x*pixel_to_mm
    
   
    
    
    for column in x:
        y = pd.concat([df.Gantry, x[column]], axis=1)
        y = y.dropna(how='any')
        c = np.polyfit(y.Gantry, y[column], 5)
        # x2g['fit'] = c[0]*x2g['Gantry'].pow(3)+c[1]*x2g['Gantry'].pow(2)+c[2]*x2g['Gantry']+c[3]
        fitline = np.poly1d(c)
        x.loc[:,column] = fitline(df.Gantry)
    
    
    x = x.join(df.Gantry)
    
    for column in range(0,33):
        
        ax = df.plot(x='Gantry', y=column, kind='scatter')
        x.plot(x='Gantry', y=column, kind='scatter', color="red", ax=ax)
        plt.show()
        
       
    plt.close("all")
    # # dfxbool = (dfx)
    # # dfx = dfx.mask(~xbool)
    
    
   
    
def polyfit_remove_outliers(df):
    """
    Update the dataframe with interpolated values for x and y positions.

    Parameters
    ----------
    df : pandas dataframe
        the dataframe with positional data for balls and apetures.

    Returns
    -------
    None.

    """
    
    model = sm.OLS(df.loci[]:,1],df.Gantry).fit()