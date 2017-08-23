    # -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:14:03 2017

@author: Brian
"""

import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)
pd.set_option('max_colwidth', 20)

class reccomender():

    def __init__(self):
        self.users = pd.read_csv('BX-Users.csv', sep = ';', quotechar = '"')
        self.ratings = pd.read_csv('ratingsclean.csv', index_col = 0)
        num = self.ratings.groupby('userID').size()
        num = num[num>5]
        filtered = self.ratings[self.ratings['userID'].isin(num.index)]
        self.books = pd.read_csv('booksclean.csv', index_col = 0)
        self.userID_u = sorted(list(filtered.userID.unique()))
        self.ISBN_u = sorted(list(filtered.ISBN.unique()))
        
        data = (filtered['rating']-filtered['rating'].mean()).tolist()
        row = filtered.userID.astype('category',
                                         categories=self.userID_u).cat.codes
        col = filtered.ISBN.astype('category',
                                       categories=self.ISBN_u).cat.codes
        self.sparse_matrix = coo_matrix((data, (row, col)), 
                                        shape=(len(self.userID_u), 
                                               len(self.ISBN_u)))
       
        
        print 'everything loaded'
    
    def SVD(self, n_components = 2):
        self.svd = TruncatedSVD(n_components = n_components)
        self.decomp = self.svd.fit_transform(self.sparse_matrix)
    
    def buildKMeans(self, n_clusters= 20):

        kmeans = KMeans(n_clusters=n_clusters).fit(self.decomp)
        groups = kmeans.labels_
        
        self.labels = pd.DataFrame({'userID':self.userID_u, 'group':groups})
        print self.labels.group.value_counts()
    

    
    def GroupRec(self, groupID, limit = 10):
        groupratings = self.ratings[self.ratings.userID.isin(self.labels[self.labels.group == groupID].userID)]
        sumratings = groupratings['rating'].groupby(groupratings['ISBN']).sum()
        numratings = groupratings['rating'].groupby(groupratings['ISBN']).size()
        dummy = 4
        recs = (sumratings+5.5*dummy)/(numratings+dummy)
        if recs.size < limit:
            limit = recs.size
                
            
        recs = recs.sort_values(ascending = False)
            
        for book in recs.index[:limit]:
            try:
                thing = self.books[self.books.ISBN == book]
                print(thing.iloc[0]['Title'] + '|' + thing.iloc[0]['Author'] + '|%g' % round(recs.ix[book], 3))
            except:
                print book
                
    def visualize(self):
        visu = pd.DataFrame({'x':rec.decomp[:,0],
                             'y':rec.decomp[:,1],
                             'group':rec.labels.group})
        sns.lmplot('x', 'y', data = visu, fit_reg = False, hue = 'group')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        
    

rec = reccomender()
rec.SVD(5)
