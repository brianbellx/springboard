# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 11:54:00 2017

@author: Brian
"""
from __future__ import division

import pandas as pd

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)
pd.set_option('max_colwidth', 20)

#users = pd.read_csv('BX-Users.csv', sep = ';', quotechar = '"')
ratings = pd.read_csv('ratingsclean.csv', index_col = 0)
books = pd.read_csv('booksclean.csv', index_col = 0, encoding = 'utf-8',
                    dtype={ 'Author': unicode,
                            'Title': unicode,
                            'ISBN':unicode,
                            'Publisher':unicode,
                            'Year': unicode,
                            'Description':unicode})
books = books.set_index('ISBN')

#list of books this user has rated
#dataframe of all reviews of those books
def bookrec(ISBN):
    print books.ix[ISBN]
    also_have_read = ratings[ratings.ISBN == ISBN]
    if also_have_read.empty:
        return 'No Matches Found'
    #dataframe of all reviews made by users who have read any of booksread
    simusers = ratings[ratings.userID.isin(also_have_read.userID)]
    
    bookmat = simusers.pivot_table(index = 'userID', 
                                   columns = 'ISBN',
                                   values = 'rating')
    
    
    bookvec = bookmat[ISBN]
    bookmat.drop(ISBN, axis = 1, inplace = True)
    dummy = 5
    
    recs = (bookmat.multiply(bookvec, axis = 0).sum()+5.5*dummy)/(bookmat.notnull().multiply(bookvec, axis = 0).sum()+dummy)
    
    limit = 10
    if recs.size < limit:
        limit = recs.size
        
    
    recs = recs.sort_values(ascending = False)
    
    for book in recs.index[:limit]:
        try:
            thing = books.ix[book]
            print(thing.Title + '|' + thing.Author + '|%g' % round(recs.ix[book], 3))
        except:
            print 'Book Not Found'