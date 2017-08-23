# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 22:46:21 2017

@author: Brian
"""
from __future__ import division

import pandas as pd


pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)
pd.set_option('max_colwidth', 20)

users = pd.read_csv('BX-Users.csv', sep = ';', quotechar = '"')
ratings = pd.read_csv('ratingsclean.csv', index_col = 0)
books = pd.read_csv('booksclean.csv', index_col = 0, encoding = 'utf-8',
                    dtype={ 'Author': unicode,
                            'Title': unicode,
                            'ISBN':unicode,
                            'Publisher':unicode,
                            'Year': unicode,
                            'Description':unicode})
books = books.set_index('ISBN')

def userrecs(userID):
    #list of books this user has rated
    booksrated = ratings[ratings.userID == userID].ISBN.tolist()

    #dataframe of all reviews of those books
    also_have_read = ratings[ratings.ISBN.isin(booksrated)]
    if len(also_have_read.userID.unique()) < 2:
        return 'No Matches Found'
    #dataframe of all reviews made by users who have read any of booksread
    simusers = ratings[ratings.userID.isin(also_have_read.userID)]
    bookmat = simusers.pivot_table(index = 'userID', 
                                   columns = 'ISBN',
                                   values = 'rating')
    uservec = bookmat.ix[userID]
    bookmat = bookmat.drop(userID)
    decay = .95
    dummy = 4
    #calculates a similarity score to our user
    
    similarity = (bookmat.subtract(uservec)**2).apply(lambda x: (decay**x)).sum(axis = 'columns')
    #removes already read books from recommendation matrix
    bookmat.drop(booksrated, axis = 1, inplace = True)
    
    
    recs = (similarity.dot(bookmat.fillna(0))+5.5*dummy)/(bookmat.notnull().multiply(similarity, axis = 'index').sum()+dummy)
    
    recs = recs.sort_values(ascending = False)
    limit = 10
    if recs.size < limit:
        limit = recs.size
        
    for book in recs.index[:limit]:
        try:
            thing = books.ix[book]
            print(thing.Title + '|' + thing.Author + '|%g' % round(recs.ix[book], 3))
        except:
            print 'Book Not Found'
    