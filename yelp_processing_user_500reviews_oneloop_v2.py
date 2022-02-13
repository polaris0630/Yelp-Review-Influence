# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:29:49 2020

@author: Yichen Jiang
"""

"""
--- this file is for extracting text features for each business in a loop ---
--- !!! based on revised functions ---
--- !!! AND perform on new yelp dataset ---

"""

# In[]
import csv
from textblob import TextBlob
from nltk.stem.snowball import EnglishStemmer
import nltk
import os
import json
from string import punctuation
import string
import numpy as np
from tqdm import tqdm
import math
from collections import Counter
import pandas as pd
import datetime
import re
import sys

# In[]
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'
sys.path.append(path)
from yelp_text_features import *

# In[]
"""
--- import business data ---

"""

path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

path_data = os.path.join(path,'yelp_dataset','business with over 500 reviews & 100 reviews per year','businesses with over 500 reviews')
path_result = os.path.join(path, 'yelp_dataset','business with over 500 reviews & 100 reviews per year','businesses with over 500 reviews with text features')

filenames_data = os.listdir(path_data)
filenames_result = os.listdir(path_result)

# In[]:
"""
--- prepare for VSM representation ---

"""
tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
stemmer = EnglishStemmer()

# read stopword list
list_stopwords = []
for line in open(os.path.join(path, 'Stopwords.txt'),"r"):
    list_stopwords.append(line)

SW = []
for i in range(len(list_stopwords)):
    if i % 2 == 0:
        SW.append(list_stopwords[i].replace('\n',''))

list_stopwords.clear()
list_stopwords = SW

del SW

# In[]:

"""
--- main process ---

"""

for filename in tqdm(filenames_data):
    """ --- check if current business has been processed already ''' """
    ## update filenames_result list
    filenames_result = os.listdir(path_result)
    business_id = filename.split('.')[0]
    if str(business_id)+'_with all features.json' in filenames_result:
        continue
    """ --- else keep going on --- """
    with open(os.path.join(path_data,filename),'r',encoding='utf-8') as file:
        for line in file.readlines():
            dict_business = json.loads(line)
    print('current business is: ', str(business_id))
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))   
    
    """ --- preprocessing --- """
    preprocessing(dict_business,stemmer, list_stopwords)
    
    """ --- calculate total term frequency --- """
    dict_wordcount = {}
    total_term_frequency(dict_wordcount, dict_business)
    
    """ --- calculate document frequency --- """
    document_frequency(dict_wordcount, dict_business)

    """ --- calculate inverse document frequency --- """
    inverse_document_frequency(dict_wordcount, dict_business)
    
    """ --- remove the top 50 and DF<25 unigrams regarding DF --- """
    # get dictionary of DF
    dict_wordcount_DF = {}
    for word in dict_wordcount.keys():
        dict_wordcount_DF[word] = dict_wordcount[word]['Document Frequency']
    # get the list of unigrams regarding DF in descending order
    list_wordcount_DF = sorted(dict_wordcount_DF.items(),key=lambda x:x[1],reverse=True)
    # get index of 1st word with DF=25
    length = 0 
    for word in list_wordcount_DF:
        if word[1] < 25:
            length = list_wordcount_DF.index(word)
            break
    # remove top 50 and DF<25 to obtain the controlled vocabulary
    list_wordcount_DF = list_wordcount_DF[50:length]
    
    """ --- construct dictionary for controlled vocabulary --- """
    dict_ctrlvoc_TF = {}
    # count term frequency for each document
    for review_id in dict_business['Reviews'].keys():
        dict_ctrlvoc_TF[review_id] = {}
        for word in list_wordcount_DF:
            dict_ctrlvoc_TF[review_id][word[0]] = dict_business['Reviews'][review_id]['text_preprocessed'].count(word[0])
    
    """ --- convert dictionary into dataframe --- """
    df_ctrlvoc_TF = pd.DataFrame.from_dict(dict_ctrlvoc_TF).T.astype(np.float64)
    
    """ --- dataframe for saving probabilities of term frequencies --- """
    df_ctrlvoc_prob = df_ctrlvoc_TF.astype(np.float64)

    """ --- calculate probability for each word --- """
    average_term_probability(df_ctrlvoc_prob,df_ctrlvoc_TF,dict_wordcount)
    
    """ --- dataframe for saving average TF-IDF weighting --- """
    df_ctrlvoc_weight = df_ctrlvoc_TF.astype(np.float64)
    
    """ --- calculate average TF-IDF weighting --- """
    # weight(term, document) = TF(term, document) * IDF(term)
    average_weighting(df_ctrlvoc_weight,df_ctrlvoc_TF,dict_wordcount)
    
    """ --- save two new variables into dict_business --- """
    """ --- extract sentiment features as well --- """
    for review_id in tqdm(df_ctrlvoc_TF.mean(axis=1).index):
        # save mean probability of TF/TTF
        dict_business['Reviews'][review_id]['mean_prob'] = df_ctrlvoc_TF.mean(axis=1)[review_id]
        # save weight = TF*IDF
        dict_business['Reviews'][review_id]['mean_weight'] = df_ctrlvoc_weight.mean(axis=1)[review_id]
        # sentiment analysis
        dict_business['Reviews'][review_id]['sentiment'] = TextBlob(dict_business['Reviews'][review_id]['text']).sentiment
        dict_business['Reviews'][review_id]['sentiment_polarity'] = dict_business['Reviews'][review_id]['sentiment'][0]
        dict_business['Reviews'][review_id]['sentiment_subjectivity'] = dict_business['Reviews'][review_id]['sentiment'][1]

    """ --- save and export dictionary in .json --- """
    with open(os.path.join(path_result, str(business_id)+'_with all features.json'), 'w+', encoding="utf-8") as outfile:
        json.dump(dict_business, outfile, ensure_ascii=False) 
    
    print(str(business_id),' has been finished')
    print('\n')
