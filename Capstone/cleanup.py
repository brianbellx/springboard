# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:26:15 2017

@author: Brian
"""

import pandas as pd
import numpy as nm
pd.set_option('display.width',120)

cl = pd.DataFrame.from_csv('processed.cleveland.data', index_col=None)
va = pd.DataFrame.from_csv('processed.va.data', index_col=None)
sw = pd.DataFrame.from_csv('processed.switzerland.data', index_col=None)
hn = pd.DataFrame.from_csv('processed.hungarian.data', index_col=None)

df = cl.append(va, ignore_index=True).append(sw, ignore_index=True).append(hn, ignore_index=True)
df.replace('?', nm.nan, inplace = True)
df.pred.replace([2,3,4], 1, inplace = True)

print df.isnull().sum().sum()
print ' '
print df.isnull().any(axis=1).value_counts()
print ' '
print df.isnull().sum()

