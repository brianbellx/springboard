# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:07:10 2017

@author: Brian
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)


hd = pd.DataFrame.from_csv('clean.csv')
yobs = hd.pred

nonbin = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
nonbindf = hd[nonbin]
scale = preprocessing.StandardScaler(with_mean = False)

nonbinstand = pd.DataFrame(scale.fit_transform(nonbindf), columns = nonbin)
#nonbinstandsq = nonbinstand.join(nonbinstand.apply(np.square),rsuffix = 'sq')
poly = PolynomialFeatures(3)
droppednonbin = hd.drop(nonbin, axis = 1)
hdstand = nonbinstand.join(droppednonbin).drop('pred', axis = 1)
hdpoly = pd.DataFrame(poly.fit_transform(hdstand))


xtrain, xtest, ytrain, ytest = train_test_split(hdpoly, yobs, train_size=.7)

clf = LogisticRegression()
gs = GridSearchCV(clf, 
                  param_grid= {"C":[.00000001, .000001,.00001,.001,.01,0.1,1]},
                  cv=5)
gs.fit(xtrain, ytrain)
bestc =  gs.best_params_['C']


clf = LogisticRegression(C = bestc)
clf.fit(xtrain, ytrain)
trainscore =clf.score(xtrain, ytrain)
testscore = clf.score(xtest, ytest)

print 'Train set accuracy: ', trainscore
print 'Test set accuracy: ', testscore

guesstest = pd.DataFrame({'prob': clf.predict_proba(xtest)[:,1], 
                   'state': clf.predict(xtest)})
acttest = ytest.reset_index()
righttest = guesstest.state ==  acttest.pred
wrongtest = ~righttest

guesstrain = pd.DataFrame({'prob': clf.predict_proba(xtrain)[:,1], 
                   'state': clf.predict(xtrain)})

acttrain = ytrain.reset_index()
righttrain = guesstrain.state ==  acttrain.pred
wrongtrain = ~righttrain

f, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.despine(left=True)
sns.distplot(guesstrain.prob[righttrain], ax = axes[0,0], kde = False, bins = 10)
sns.distplot(guesstrain.prob[wrongtrain], ax = axes[0,1], kde = False, bins = 10)
sns.distplot(guesstest.prob[righttest], ax = axes[1,0], kde = False, bins = 10)
sns.distplot(guesstest.prob[wrongtest], ax = axes[1,1], kde = False, bins = 10)

cmtrain = confusion_matrix(ytrain, guesstrain.state)
cmtest = confusion_matrix(ytest, guesstest.state)
print cmtrain
print cmtest