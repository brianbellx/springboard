# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:40:46 2017

@author: Brian
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import preprocessing


pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)


hd = pd.DataFrame.from_csv('clean.csv')
hdnorm = pd.DataFrame(preprocessing.normalize(hd), columns = hd.columns)
yobs = hd.pred
xobs = hd.drop('pred', axis = 1)
xobsnorm = hdnorm.drop('pred', axis = 1)

xtrain, xtest, ytrain, ytest = train_test_split(xobs, yobs, train_size=.7)

clf = LogisticRegression()
gs = GridSearchCV(clf, 
                  param_grid= {"C": [.001, .01 , 0.1, 1, 10, 100], },cv=5)
gs.fit(xtrain, ytrain)
bestc =  gs.best_params_['C']

clf = LogisticRegression(C = bestc)
clf.fit(xtrain, ytrain)
trainscore =clf.score(xtrain, ytrain)
testscore = clf.score(xtest, ytest)

print 'Train set accuracy: ', trainscore
print 'Test set accuracy: ', testscore
