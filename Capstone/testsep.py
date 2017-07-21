# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:50:13 2017

@author: Brian
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Load in cleaned data after runnning cleanup.py
hd = pd.DataFrame.from_csv('clean.csv')

#create series for true values of predicted attribute
yobs = hd.pred

#create a separae dataframe for numerical, non binary data
nonbin = ['age', 'trestbps', 'chol', 'thalach', 'ca']
nonbindf = hd[nonbin]

#scale nunmerical data down to 0-1 by dividing by max
nonbinscaled = nonbindf/nonbindf.max()

#add columns for square of numerical data
nonbinscaledsq = nonbinscaled.join(nonbinscaled.apply(np.square),rsuffix = 'sq')

#remove the old numerical data and replace with standardized and squared data
droppednonbin = hd.drop(nonbin, axis = 1)
hdstand = nonbinscaledsq.join(droppednonbin).drop('pred', axis = 1)


#separate by test category
hdbasic = hdstand[['age', 'agesq', 'sex',  
                   'trestbps','trestbpssq',
                   'cp_1', 'cp_2', 'cp_3', 'cp_4',]]
hdrestecg = hdstand[[ 'restecg_0', 'restecg_1', 'restecg_2']]
hdlabwork = hdstand[['chol','cholsq', 'fbs']]
hdecg = hdstand[['exang', 'oldpeak',
                       'slope_1', 'slope_2', 'slope_3' ]]
hdthal = hdstand[['thalach','thalachsq',
                  'thal_3','thal_6','thal_7']]
hdca = hdstand[['ca', 'casq']]
hdall = hdbasic.join(hdlabwork).join(hdecg).join(hdthal).join(hdca)

#defining a function so I can combine the above dataframes on the fly
def doit(data, labels):
    #train test split with 70% going to train
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, train_size=.7)

    #gridsearch to optimize C
    clf = LogisticRegression()
    gs = GridSearchCV(clf, 
                  param_grid= {"C":[.001,.01,0.1,1,10]},
                  cv=5)
    gs.fit(xtrain, ytrain)
    bestc =  gs.best_params_['C']
    print bestc

    #recreate the classifier with new regularization parameter
    clf = LogisticRegression(C = bestc)
    clf.fit(xtrain, ytrain)
    #score on train and test sets
    trainscore =clf.score(xtrain, ytrain)
    testscore = clf.score(xtest, ytest)

    print 'Train set accuracy: ', trainscore
    print 'Test set accuracy: ', testscore

    #new dataframe with the probability and prediction
    guesstest = pd.DataFrame({'prob': clf.predict_proba(xtest)[:,1], 
                   'state': clf.predict(xtest)})
    #create a mask for which guesses were incorrect and correct
    acttest = ytest.reset_index()
    righttest = guesstest.state ==  acttest.pred
    wrongtest = ~righttest

    #plots histograms of probability scores for right and wrong guesses
    sns.set_palette("dark")
    sns.distplot(guesstest.prob[righttest],kde = False, bins = 10, label = 'Correct Prediction')
    sns.distplot(guesstest.prob[wrongtest],kde = False, bins = 10, label = 'Incorrect Prediction')
    plt.legend()
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    #print a confusion matrix
    cmtest = confusion_matrix(ytest, guesstest.state)
    print cmtest
    #return the coefficients to see feature importance
    return pd.Series(clf.coef_.flatten(), index = data.columns)


