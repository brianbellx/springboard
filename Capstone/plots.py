# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:06:46 2017

@author: Brian
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)

hd = pd.DataFrame.from_csv('clean.csv', index_col=0)

hdpos = hd[hd.pred == 1]
hdneg = hd[hd.pred == 0]

cr = hd.corr()

sns.heatmap(cr)



'''f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
plt.xlim(0,400)
sns.despine(left=True)
sns.distplot(hdpos.chol, ax = axes[0,0])
sns.distplot(hdpos.trestbps, ax = axes[0,1])
sns.distplot(hdneg.chol, ax = axes[1,0])
sns.distplot(hdneg.trestbps, ax = axes[1,1])'''

hdcp = hd[['cp_1', 'cp_2', 'cp_3', 'cp_4', 'pred']]
hdcp.columns = ['Typical Angina', 'Atypical Angina', 'Non Anginal Pain', 'Asymptomatic', 'Heart Disease']
oiu =pd.melt(hdcp, 
             id_vars = 'Heart Disease',
             value_vars=['Typical Angina','Atypical Angina',
                               'Non Anginal Pain','Asymptomatic'],
                                var_name='Chest Pain Type')
sieve = oiu[oiu['value'] == 1]
g = sns.factorplot(x = 'Chest Pain Type', data = sieve, kind = 'count', hue = 'Heart Disease')