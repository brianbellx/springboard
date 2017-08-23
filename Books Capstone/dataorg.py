# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:38:33 2017

@author: Brian
"""

import pandas as pd
from BeautifulSoup import BeautifulSoup

def decoded(df, col):
   return df[col].str.decode('utf-8', errors = 'ignore')

pd.set_option('display.width',120)
pd.set_option('display.max_columns', 25)
pd.set_option('max_colwidth', 20)

users = pd.read_csv('BX-Users.csv', sep = ';', quotechar = '"')
books = pd.read_csv('BX-books.csv', sep = ';', quotechar = '"', escapechar = '\\')
books.columns = [u'ISBN', u'BookTitle', u'BookAuthor', u'YearOfPublication',
       u'Publisher', u'ImageURLS', u'ImageURLM', u'ImageURLL']
books = books[[u'ISBN', u'BookTitle', u'BookAuthor', 
                  u'YearOfPublication', u'Publisher']]
ratings = pd.read_csv('ratingsclean.csv', index_col = 0)
details = pd.read_csv('details.csv', index_col = 0)
details2 = pd.read_csv('descriptions2.csv', index_col = 0)
details3 = pd.read_csv('descriptions3.csv', index_col = 0)

details.columns = ['description', 'ISBN', 'BookTitle']

books1 = books.merge(details, how = 'outer', left_on = 'ISBN', right_on='ISBN')
books1 = books1[[u'ISBN',  u'BookTitle_y', u'BookAuthor',
                 u'YearOfPublication', u'Publisher', u'description']]
books1.columns = [u'ISBN',  u'BookTitle', u'BookAuthor',
                 u'YearOfPublication', u'Publisher', u'description']

details2.BookAuthor = details2.BookAuthor.apply(lambda x: x[1:-1])
details2.YearOfPublication = details2.YearOfPublication.apply(lambda x: x[-5:-1])
details3.YearOfPublication = details3.YearOfPublication.apply(lambda x: x[-5:-1])



books2 = books1.append(details2).append(details3)
books2.columns = [u'Author',  u'Title', u'ISBN',
                 u'Publisher', u'Year', u'Description']
#books2 = books2.reset_index()
books2.Description = books2.Description.astype('str').apply(lambda x: x.decode('utf-8'))
books2.Description = books2.Description.apply(lambda x: BeautifulSoup(x).text)
books2.Author = decoded(books2, 'Author')
books2.Title = decoded(books2, 'Title')
books2.Publisher = decoded(books2, 'Publisher')


books2.to_csv('booksclean.csv', encoding = 'utf-8')