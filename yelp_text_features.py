# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:26:09 2019

@author: Yichen Jiang
"""

"""
--- this file is for saving functions for extracting text features ---

"""

from textblob import TextBlob
from nltk.stem.snowball import EnglishStemmer
import nltk
import string
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

# In[]
"""
--- define preprocessing function ---

"""
def preprocessing(dict_business,stemmer,list_stopwords):
    
    list_temp = []
    
    for review_id in dict_business['Reviews'].keys():
        # Tokenization
        #text = tokenizer.tokenize(dict_business['Reviews'][review_id]['text'])
        text = TextBlob(dict_business['Reviews'][review_id]['text']).words
        #dict_business['Reviews'][review_id]
        # Stemming
        for word in text:
            word = stemmer.stem(word)
            # Punctuation removal
            word_clean = ""
            for elem in word:
                if elem not in string.punctuation:
                    word_clean += elem
            if word_clean.isdecimal():
                word_clean = "NUM"
            # Stopwords removal
            if word_clean not in list_stopwords:
                list_temp.append(word_clean)
            
        dict_business['Reviews'][review_id]['text_preprocessed'] = list_temp
        dict_business['Reviews'][review_id]['textclean_length'] = len(list_temp)
        dict_business['Reviews'][review_id]['text_tokenization'] = text
        dict_business['Reviews'][review_id]['text_length'] = len(text)
        
        list_temp = []

# In[]
"""
--- define function for calculating total term frequency ---

"""
def total_term_frequency(dict_wordcount, dict_business):

    for review_id in dict_business['Reviews'].keys():
        for word in dict_business['Reviews'][review_id]['text_preprocessed']:
            if word not in dict_wordcount:
                dict_wordcount[word] = {'Total Term Frequency': 1}
            else:
                dict_wordcount[word]['Total Term Frequency'] += 1

# In[]:
"""
--- define function for calculating document frequency ---

"""
def document_frequency(dict_wordcount, dict_business):  

    for word in dict_wordcount.keys():
        for review_id in dict_business['Reviews'].keys():
            if word in dict_business['Reviews'][review_id]['text_preprocessed']:
                if 'Document Frequency' not in dict_wordcount[word].keys():
                    dict_wordcount[word]['Document Frequency'] = 1
                else:
                    dict_wordcount[word]['Document Frequency'] += 1

# In[]
"""
--- define function for calculating inverse document frequency ---

"""
def inverse_document_frequency(dict_wordcount, dict_business):

    for word in dict_wordcount.keys():
        DF = dict_wordcount[word]['Document Frequency']
        IDF = 1 + math.log(len(dict_business['Reviews'])/DF)
        dict_wordcount[word]['Inverse Document Frequency'] = IDF    

# In[]
"""
--- define function for calculating average term probability

"""
def average_term_probability(df_ctrlvoc_prob,df_ctrlvoc_TF,dict_wordcount):
    # for each column (token)
    for token in df_ctrlvoc_prob.columns.values.tolist():
        # obtain total term frequency for current column(token) from dictionary
        TTF = dict_wordcount[token]['Total Term Frequency']
        # calculate probability: TF(d)/TTF
        df_ctrlvoc_prob[token] = df_ctrlvoc_TF[token]/TTF

# In[]
"""
--- define function for calculating average TF-IDF weighting ---

"""
def average_weighting(df_ctrlvoc_weight,df_ctrlvoc_TF,dict_wordcount):
    for token in df_ctrlvoc_weight.columns.values.tolist():
        # obtain inverse document frequency for current column(token) from dictionary
        IDF = dict_wordcount[token]['Inverse Document Frequency']
        # calculate weight(t,d) = TF(t,d) * IDF(t)
        df_ctrlvoc_weight[token] = df_ctrlvoc_TF[token] * IDF

# In[]
