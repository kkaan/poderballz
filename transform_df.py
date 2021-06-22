# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:37:55 2021

@author: 56153805
"""
#!/usr/bin/env python3
import pickle
import pandas as pd
import itertools
import numpy as np

data = pd.DataFrame(np.random.randn(10, 5), columns=('0_n', '1_n', '0_p', '1_p', 'x'))

indices = set()
groups = set()
others = set()
for c in data.columns:
    if '_' in c:
        (i, g) = c.split('_')
        c2 = pd.MultiIndex.from_tuples((i, g),)
        indices.add(int(i))
        groups.add(g)
    else:
        others.add(c)
columns = list(itertools.product(groups, indices))
columns = pd.MultiIndex.from_tuples(columns)
ret = pd.DataFrame(columns=columns)
for c in columns:
    ret[c] = data['%d_%s' % (int(c[1]), c[0])]
for c in others:
    ret[c] = data['%s' % c]
ret.rename(columns={'total': 'total_indices'}, inplace=True)

print("Before:")
print(data)
print("")
print("After:")
print(ret)