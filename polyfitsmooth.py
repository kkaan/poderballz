# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:27:47 2021

@author: 56153805
"""

import numpy as np
import pandas as pd


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
