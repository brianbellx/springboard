# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 08:24:01 2017

@author: Brian
"""

import pandas as pd
from pprint import pprint
from goodreads import client

def decoded(df, col):
   return df[col].str.decode('utf-8', errors = 'ignore')

gc = client.GoodreadsClient('oJeAwNDiKjlnquHAoYSgw', 'BJ02lX5ACiQo0FKqKsEROty38LoixayzWNvXWv0E9Ow')

books = pd.read_csv('BX-books.csv', sep = ';', quotechar = '"', escapechar = '\\')
books.columns = [u'ISBN', u'BookTitle', u'BookAuthor', u'YearOfPublication',
       u'Publisher', u'ImageURLS', u'ImageURLM', u'ImageURLL']
books = books[[u'ISBN', u'BookTitle', u'BookAuthor', 
                  u'YearOfPublication', u'Publisher']]
details = pd.read_csv('details.csv', index_col = 0)
books.ISBN = decoded(books, 'ISBN')

ratings = pd.read_csv('ratingsclean.csv', index_col = 0)
ratings.ISBN = decoded(ratings, 'ISBN')
found = pd.read_csv('descriptions2.csv', index_col = 0)
books = books.append(found)

ratings.ISBN = ratings.ISBN.apply(lambda x: ''.join(c for c in x if c not in 'ISBN:*"\\ ./()$=?<>&#'))

ratings = ratings.drop(ratings[(ratings.ISBN.str.len()<8)].index)
mask = ((ratings.ISBN.str.len() == 8) | (ratings.ISBN.str.len() == 11))
ratings.loc[mask, 'ISBN'] = '0' + ratings[mask].ISBN
mask = ((ratings.ISBN.str.len() == 9) | (ratings.ISBN.str.len() == 12))
ratings.loc[mask, 'ISBN'] = '0' + ratings[mask].ISBN

notin = ratings[~ratings.ISBN.isin(books.ISBN)]


setnotin = notin.ISBN.unique()
print len(setnotin)
gcdesc = []

for idx, ISBN in enumerate(setnotin):
    try:
        gcbook = gc.book(isbn = ISBN)
    except:
        pprint(str(idx  ) + ':ERROR')
        continue
    entry = {'ISBN':ISBN}
    try:
        entry['BookTitle']= gcbook.title
    except:
        pass
    try:
        entry['description']=gcbook.description
    except:
        pass
    try:
        entry['BookAuthor']=gcbook.authors[0]
    except:
        pass
    try:
        entry['YearOfPublication']= gcbook.publication_date
    except:
        pass
    try:
        entry['Publisher']=gcbook.publisher
    except:
        pass
    pprint(str(idx)+ ':' + entry['BookTitle'])
    gcdesc.append(entry)

descriptions = pd.DataFrame(gcdesc)