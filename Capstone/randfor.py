# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:28:32 2017

@author: Brian
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
##import numpy as np
##import seaborn as sns

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)


hd = pd.DataFrame.from_csv('clean.csv')

yobs = hd.pred
xobs = hd.drop('pred', axis = 1)

nonbin = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
nonbindf = xobs[nonbin]

nonbinscaled = nonbindf/nonbindf.max()
nonbinscaledsq = nonbinscaled.join(nonbinscaled.apply(np.square),rsuffix = 'sq')
droppednonbin = hd.drop(nonbin, axis = 1)
hdstand = nonbinscaledsq.join(droppednonbin).drop('pred', axis = 1)


#separate by test category
hdbasic = hdstand[['age', 'agesq', 'sex',  
                   'trestbps','trestbpssq',
                   'cp_1', 'cp_2', 'cp_3', 'cp_4',
              'restecg_0', 'restecg_1', 'restecg_2']]
hdlabwork = hdstand[['chol','cholsq', 'fbs']]
hdecg = hdstand[['exang', 'oldpeak',
                       'slope_1', 'slope_2', 'slope_3' ]]
hdthal = hdstand[['thalach','thalachsq',
                  'thal_3','thal_6','thal_7']]
hdca = hdstand[['ca', 'casq']]

hdall = hdbasic.join(hdlabwork).join(hdecg).join(hdthal).join(hdca)


def dorf(data, labels):
    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, train_size=.7,)

    clf = RandomForestClassifier()
    params = {'max_features' :  range(1, 5), 'min_samples_split': range(15,35) }
    gs = GridSearchCV(clf, param_grid= params,cv=5)
    gs.fit(xtrain, ytrain)
    bestmf =  gs.best_params_['max_features']
    bestmss = gs.best_params_['min_samples_split']
    clf = RandomForestClassifier(n_estimators = 100,
                             max_depth = 20,
                             min_samples_split = bestmss,
                             max_features= bestmf)
    print bestmf
    print bestmss
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
    return pd.Series(clf.feature_importances_, index = data.columns)
