# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:48:51 2021

Y = dYcos(G)
x = dXsin(G)
z = dY

@author: 56153805
"""

import numpy as np

def roomcoords(df):
    
    df_room = df.loc['WL'].copy()
    df_room.loc[:,'x'] = df.loc['WL']*np.sin(df.loc[:,'Gantry']*np.pi/180)