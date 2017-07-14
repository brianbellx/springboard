# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:52:35 2017

@author: Brian
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
##import numpy as np
##import seaborn as sns

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)


hd = pd.DataFrame.from_csv('clean.csv')

hdselect = hd[['thalach', 'oldpeak', 'ca', 'sex']]

yobs = hd.pred
xobs = hdselect

xtrain, xtest, ytrain, ytest = train_test_split(xobs, yobs, train_size=.7,
                                                random_state = 13)

clf = LogisticRegression()
gs = GridSearchCV(clf, 
                  param_grid= {"C": [.001, .01 , 0.1, 1, 10, 100], },cv=5)
gs.fit(xtrain, ytrain)
bestc =  gs.best_params_['C']

clf = LogisticRegression(C = bestc)
clf.fit(xtrain, ytrain)
trainscore = accuracy_score(clf.predict(xtrain), ytrain)
testscore = accuracy_score(clf.predict(xtest), ytest)

print 'Train set accuracy: ', trainscore
print 'Test set accuracy: ', testscore
