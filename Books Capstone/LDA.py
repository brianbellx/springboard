# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:05:25 2017

@author: Brian
"""
import pandas as pd
#from langdetect import detect
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
import spacy
#import codecs
#import itertools as it
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim
#import warnings
import cPickle as pickle
import numpy as np
from ast import literal_eval
from scipy import spatial

books = pd.read_csv('booksclean.csv', index_col = 0, encoding = 'utf-8',
                    dtype={ 'Author': unicode,
                            'Title': unicode,
                            'ISBN':unicode,
                            'Publisher':unicode,
                            'Year': unicode,
                            'Description':unicode})
books = books.set_index('ISBN')
#creat dataframe of only English descriptions above 30 characters
#hasdesc = books[books.Description.str.len() > 30]
#hasdesc = hasdesc.drop(256127)

#mask = hasdesc.Description.apply(lambda x: detect(x))
#english = hasdesc[mask == 'en']
english = pd.read_csv('englishtokens.csv', index_col = 0, encoding = 'utf-8',
                      converters={'desc_vec': literal_eval,
                                  'ldavec': literal_eval})

#returns a boolean for if a token is punctuation or a space
def punct_space(token):
    return token.is_punct or token.is_space

#load the English model
nlp = spacy.load('en')
#reads through the descriptions and removes formatting from words
def lemmatized_sentence_corpus(textvec):
    for parsed_review in nlp.pipe(textvec, batch_size = 10000, n_threads = 4):
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for 
                             token in sent if not punct_space(token)])

#write the lemmatized sentences to a file    
unigram_sentences_filepath = 'unigram_sentences_all.txt'
'''with codecs.open(unigram_sentences_filepath,'w', encoding = 'utf-8') as f:
    for sentence in lemmatized_sentence_corpus(english.Description):
        f.write(sentence + '\n')
'''

unigram_sentences = LineSentence(unigram_sentences_filepath)

bigram_model_filepath = 'bigram_model_all.txt'
#build two word phrases
'''bigram_model = Phrases(unigram_sentences)
bigram_model.save(bigram_model_filepath)
'''

bigram_model = Phrases.load(bigram_model_filepath)
bigram_sentences_filepath = 'bigram_sentences_all.txt'
#write bigrammed sentences to a file
'''with codecs.open(bigram_sentences_filepath, 'w',encoding = 'utf-8') as f:
    for unigram_sentence in unigram_sentences:
        bigram_sentence = u' '.join(bigram_model[unigram_sentence])
        f.write(bigram_sentence + '\n')
'''

bigram_sentences = LineSentence(bigram_sentences_filepath)

trigram_model_filepath = 'trigram_model_all.txt'
#build three word phrases
'''trigram_model = Phrases(bigram_sentences)
trigram_model.save(trigram_model_filepath)
'''

trigram_model = Phrases.load(trigram_model_filepath)
trigram_sentences_filepath = 'trigram_sentences_all.txt'
#write trigrammed sentences to file
'''with codecs.open(trigram_sentences_filepath, 'w',encoding = 'utf-8') as f:
    for bigram_sentence in bigram_sentences:
        trigram_sentence = u' '.join(trigram_model[bigram_sentence])
        f.write(trigram_sentence + '\n')
'''

trigram_sentences = LineSentence(trigram_sentences_filepath)
    
trigram_descriptions_filepath = 'trigram_descriptions.txt'
words = []
# for each description, lemmatize, apply bigram and trigram models, and
# remove stopwords
'''with codecs.open(trigram_descriptions_filepath, 'w',encoding = 'utf-8') as f:

    for parsed_review in nlp.pipe(english.Description,
                              batch_size = 10000, n_threads = 4):
        unigram_description = [token.lemma_ for token in parsed_review
                               if not punct_space(token)]
        
        bigram_description = bigram_model[unigram_description]
        trigram_description = trigram_model[bigram_description]
        
        
        
        trigram_description = [term for term in trigram_description
                               if term not in spacy.en.STOP_WORDS]
        trigram_description = u' '.join(trigram_description)
        words.append(trigram_description)
        f.write(trigram_description+'\n')
'''

'''english['tokenized_desc'] = words
english.to_csv('englishtokens.csv', encoding = 'utf-8')
print 'Done'
'''

trigram_descriptions = LineSentence(trigram_descriptions_filepath)
trigram_dict_filepath = 'trigram_dict'

# build dictionary
trigram_dict = Dictionary(trigram_descriptions)
#remove common or exceedingly rare terms
'''trigram_dict.filter_extremes(no_below = 10, no_above = .4)
trigram_dict.compactify()

trigram_dict.save(trigram_dict_filepath)
trigram_dict = Dictionary.load(trigram_dict_filepath)
'''
#format data for use in LDA
'''trigram_bow_filepath = 'trigram_bow_corpus_all.mm'
def trigram_bow_generator(filepath):
    for description in LineSentence(filepath):
        yield trigram_dict.doc2bow(description)'''

'''MmCorpus.serialize(trigram_bow_filepath,
                   trigram_bow_generator(trigram_descriptions_filepath))'''

#trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

lda_model_filepath = 'lda_model_all'
# learns the LDA topics
'''with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lda = LdaMulticore(trigram_bow_corpus, num_topics=50,
                       id2word = trigram_dict, workers = 3)
    
lda.save(lda_model_filepath)'''

lda = LdaMulticore.load(lda_model_filepath)
#visualize LDA topics
LDAvis_filepath = 'LDAvis_prep'
'''LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus, trigram_dict)
with open(LDAvis_filepath, 'w') as f:
    pickle.dump(LDAvis_prepared, f)'''

with open(LDAvis_filepath) as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.display(LDAvis_prepared)

from gensim.models import Word2Vec
trigram_sentences = LineSentence(trigram_sentences_filepath)
word2vec_filepath = 'word2vec_model_all'
# learn word-vectors with 12 passes over descriptions
'''w2v = Word2Vec(trigram_sentences, size = 100, window = 5, min_count = 20, sg=1, workers = 4)
w2v.save(word2vec_filepath)
for i in range(1,12):
    print i
    w2v.train(trigram_sentences, total_examples=w2v.corpus_count, epochs=w2v.iter)
    w2v.save(word2vec_filepath)'''
    
w2v = Word2Vec.load(word2vec_filepath)
w2v.init_sims()

#takes a block of text and removes formatting
def tokenize(text):
    parsed_text = nlp(text)
    uni_text = [token.lemma_ for token in parsed_text if not punct_space(token)]
    bi_text = bigram_model[uni_text]
    tri_text = trigram_model[bi_text]
    
    tri_text = [term for term in tri_text if not term in spacy.en.STOP_WORDS]
    return tri_text

# creates a document vector from word vectors
def docvec(text):
    
    total = sum(w2v.wv.word_vec(token) for token in tokenize(text) if token in w2v.wv.vocab)
    return total
#desc_vec = english.Description.apply(docvec)
#english['desc_vec'] = desc_vec

#cosine similarity between two vectors
def compare(v1, v2):
    try:
        result = 1 - spatial.distance.cosine(v1, v2)
        return result
    except:
        return 0

#creates a 50 element vector from lda_topics
def lda_topics(text, min_topic_freq = 0.05):
    parsed_text = nlp(text)
    unigram_text = [token.lemma_ for token in parsed_text if not punct_space(token)]
    bigram_text = bigram_model[unigram_text]
    trigram_text = trigram_model[bigram_text]
    
    trigram_text = [term for term in trigram_text if not term in spacy.en.STOP_WORDS]
    
    text_bow = trigram_dict.doc2bow(trigram_text)
    vec = np.zeros(50)
    jk = lda[text_bow]
    for idx, value in jk:
        vec[idx] = value
    return vec
    
#ldavec = english.Description.apply(lda_topics)
#english['ldavec'] = ldavec
#english.to_csv('englishtokens.csv', encoding = 'utf-8')

#returns a sorted list of books with most similar lda vectors
def ldacomp(ISBN):
    v2 = english.set_index('ISBN').ix[ISBN].ldavec
    trial = english.ldavec.apply(lambda x: compare(v2, x))
    return trial.sort_values(ascending = False)[1:]
#returns a sorted list of books with most similar word vectors
def w2vcomp(ISBN):
    v1 = english.set_index('ISBN').ix[ISBN].desc_vec
    trial = english.desc_vec.apply(lambda x: compare(v1, x))
    return trial.sort_values(ascending = False)[1:]
#returns a sorted list of books most similar to the inputted text
def w2vcomptext(text):
    v1 = docvec(text)
    trial = english.desc_vec.apply(lambda x: compare(v1, x))
    return trial.sort_values(ascending = False)


def displaysim(recs, limit=10):
    for book in recs.index[1:limit+1]:
        try:
            ISBN = english.ix[book].ISBN
            thing = books.ix[ISBN]
            print(thing.Title + '|' + thing.Author + '|%g' % round(recs[book], 3))
        except:
            print 'Book Not Found'
            
            