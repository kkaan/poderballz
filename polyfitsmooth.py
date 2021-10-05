# -*- coding: utf-8 -*-
"""
Interpolate the positional data to match the required gantry positions.

Created on Sun Oct  3 16:27:47 2021

@author: 56153805
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import pandas

def remove_outliers(df, what):
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
        
    window = 10
    num_stds = 3
    
    
    x = df[what].copy()
    
    # add values to front for rolling average to work at start.
    m = np.pad(x, pad_width=(window-1, 0), mode='wrap')
    m = pandas.Series(m)
    m = m.rolling(window).median()
    
    # crop the NAN values
    m = m.iloc[window-1:]
    m.reset_index(drop=True, inplace=True)
    
    s = np.pad(m, pad_width=(window-1, 0), mode='wrap')
    s = pandas.Series(s)
    s = s.rolling(window).std()
    
    s = s.iloc[window-1:]
    s.reset_index(drop=True, inplace=True)
    
    xbool = (x <= m+num_stds*s) & (x >= m-num_stds*s)
    
    x2 = x.copy()
    x2 = x2.mask(~xbool)


    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('window = {swin}, stdevs = {snumstds}'.format(swin = window, snumstds = num_stds))
    x2.plot(ax=ax1)
    plt.show()

def polyfit_interpolate(df):
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
    gantry = df["Gantry"].copy()
    
    
    x2g = pd.concat([gantry,x2],axis=1)
    
    x2g.dropna(inplace=True)
    
    
    c = np.polyfit(x2g['Gantry'].astype(float), x2g[('EPIDBalls', 1, 'x')].astype(float), 5)
    # x2g['fit'] = c[0]*x2g['Gantry'].pow(3)+c[1]*x2g['Gantry'].pow(2)+c[2]*x2g['Gantry']+c[3]
    fitline = np.poly1d(c)
    
    
    
    plt.plot(x2g['Gantry'], x2g[('EPIDBalls', 1, 'x')], '.')
    plt.plot(x2g['Gantry'],fitline(x2g['Gantry']))
    # plt.plot(x2g['Gantry'], x2g['fit'], label='after fitting', color='red')
    plt.legend()
    plt.show()
