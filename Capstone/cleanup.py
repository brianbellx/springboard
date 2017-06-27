# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:26:15 2017

@author: Brian
"""

import pandas as pd
import numpy as nm
from fancyimpute import KNN
from sklearn import preprocessing
pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)

#Loads in the four datasets and combines them
cl = pd.DataFrame.from_csv('processed.cleveland.data', index_col=None)
va = pd.DataFrame.from_csv('processed.va.data', index_col=None)
sw = pd.DataFrame.from_csv('processed.switzerland.data', index_col=None)
hn = pd.DataFrame.from_csv('processed.hungarian.data', index_col=None)
df = cl.append(va, ignore_index=True).append(sw, ignore_index=True).append(hn, ignore_index=True)

#replace missing information with NaN, change the response variable
# from [0,1,2,3,4] to [0,1], and change the data type to float for consistency
df.replace('?', nm.nan, inplace = True)
df.pred.replace([2,3,4], 1, inplace = True)
df = df.astype(float)
df.chol.replace(0, nm.nan, inplace = True)


#count up NaNs
print df.isnull().sum().sum()
print ' '
print df.isnull().any(axis=1).value_counts()
print ' '
print df.isnull().sum()

#creates a subset dataframe for categorical data
cat = df[['cp', 'restecg', 'slope', 'thal']]
cat.fillna(99, inplace = True)


catcols = ['cp_1', 'cp_2', 'cp_3', 'cp_4',
           'restecg_0','restecg_1','restecg_2','restecg_miss',
           'slope_1','slope_2', 'slope_3', 'slope_miss',
           'thal_3', 'thal_6', 'thal_7', 'thal_miss']

#break up categorical columns into multiple 0 or 1 columns
enc = preprocessing.OneHotEncoder()
catdf = pd.DataFrame(enc.fit_transform(cat).toarray(), columns = catcols)

#Replaces 0s with NaN for rows marked as missing data
catdf.restecg_0[catdf.restecg_miss == 1] = nm.nan
catdf.restecg_1[catdf.restecg_miss == 1] = nm.nan
catdf.restecg_2[catdf.restecg_miss == 1] = nm.nan
catdf.slope_1[catdf.slope_miss == 1] = nm.nan
catdf.slope_2[catdf.slope_miss == 1] = nm.nan
catdf.slope_3[catdf.slope_miss == 1] = nm.nan
catdf.thal_3[catdf.thal_miss == 1] = nm.nan
catdf.thal_6[catdf.thal_miss == 1] = nm.nan
catdf.thal_7[catdf.thal_miss == 1] = nm.nan
catdf.drop(['slope_miss','restecg_miss','thal_miss'],axis = 1,inplace = True)

#removes categorical data and replaces it with the above onehot encoded df
prefill = df.drop(['cp','restecg','slope','thal'], axis = 1).join(catdf)
cols = [u'age', u'sex',
             u'cp_1', u'cp_2', u'cp_3', u'cp_4',
             u'trestbps', u'chol', u'fbs',
             u'restecg_0', u'restecg_1', u'restecg_2',
             u'thalach', u'exang', u'oldpeak',
             u'slope_1', u'slope_2', u'slope_3',
             u'ca', u'thal_3', u'thal_6', u'thal_7',u'pred']
prefill = prefill[cols]

#fills NaNs by KNN imputation
fill = KNN(k=7).complete(prefill)
data = pd.DataFrame(fill, columns = cols)

print data.isnull().sum().sum()
print ' '
print data.isnull().any(axis=1).value_counts()
print ' '
print data.isnull().sum()
# Turns data back into ints, except oldpeak, which needs to stay a float
temp = data.oldpeak
data = data.round().astype(int)
data.oldpeak = temp

data.to_csv('clean.csv')