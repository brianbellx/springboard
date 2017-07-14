# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:35:38 2017

@author: Brian
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pandas as pd
##import numpy as np
##import seaborn as sns

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)


hd = pd.DataFrame.from_csv('clean.csv')

yobs = hd.pred
xobs = hd.drop('pred', axis = 1)

xtrain, xtest, ytrain, ytest = train_test_split(xobs, yobs, train_size=.6,
                                                random_state = 13)
clf = svm.SVC()
gs = GridSearchCV(clf, 
                  param_grid= {"C": [.001, .01 , 0.1, 1, 10, 100], },cv=5)
gs.fit(xtrain, ytrain)
bestc =  gs.best_params_['C']


clf = svm.SVC(C = bestc)
clf.fit(xtrain, ytrain)
trainscore =clf.score(xtrain, ytrain)
testscore = clf.score(xtest, ytest)

print 'Train set accuracy: ', trainscore
print 'Test set accuracy: ', testscore
