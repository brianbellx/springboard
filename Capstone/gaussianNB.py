# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:20:08 2017

@author: Brian
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import pandas as pd
##import numpy as np
##import seaborn as sns

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)


hd = pd.DataFrame.from_csv('clean.csv')

yobs = hd.pred
xobs = hd.drop('pred', axis = 1)

xtrain, xtest, ytrain, ytest = train_test_split(xobs, yobs, train_size=.7,)

#clf = RandomForestClassifier()
#params = {'max_features' :  range(1, 10), 'min_samples_split': range(15,35) }
#gs = GridSearchCV(clf, param_grid= params,cv=5)
#gs.fit(xtrain, ytrain)
#bestmf =  gs.best_params_['max_features']
#bestmss = gs.best_params_['min_samples_split']
clf = GaussianNB()
clf.fit(xtrain, ytrain)
trainscore =clf.score(xtrain, ytrain)
testscore = clf.score(xtest, ytest)

print 'Train set accuracy: ', trainscore
print 'Test set accuracy: ', testscore
