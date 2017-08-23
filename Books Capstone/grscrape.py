# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 11:42:39 2017

@author: Brian
"""
import pandas as pd
from pprint import pprint
from goodreads import client
gc = client.GoodreadsClient('oJeAwNDiKjlnquHAoYSgw', 'BJ02lX5ACiQo0FKqKsEROty38LoixayzWNvXWv0E9Ow')

books = pd.read_csv('BX-books.csv', sep = ';', quotechar = '"', escapechar = '\\')
books.columns = [u'ISBN', u'BookTitle', u'BookAuthor', u'YearOfPublication',
       u'Publisher', u'ImageURLS', u'ImageURLM', u'ImageURLL']
books = books[[u'ISBN', u'BookTitle', u'BookAuthor', 
                  u'YearOfPublication', u'Publisher']]

gcdesc = []

for idx, ISBN in enumerate(books.ISBN):
    try:
        gcbook = gc.book(isbn = ISBN)
    except:
        entry = {'isbn':ISBN, 'title':None, 'description':None}
        pprint(str(idx  ) + ':ERROR')
        continue
    
    entry = {'isbn':ISBN, 'title':gcbook.title, 'description':gcbook.description}
    pprint(str(idx)+ ':' + entry['title'])
    gcdesc.append(entry)

descriptions = pd.DataFrame(gcdesc)
