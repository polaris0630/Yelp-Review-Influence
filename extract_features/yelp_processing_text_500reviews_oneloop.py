# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:19:01 2019

@author: Yichen Jiang
"""

"""
--- this file is for extracting text features for each business in a loop ---

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
createVar = locals()

# In[]:
"""
--- read business list with more than 500 reviews ---

"""



path_input = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp\yelp_dataset\business with over 500 reviews & 100 reviews per year\businesses with over 500 reviews'

path_text = os.path.join(path,'business with text features')

file = open(path_input, 'r', encoding = 'utf-8')
lines = csv.reader(file)

list_business_id = []
for line in lines:
    if line[3] == 'business_id':
        continue
    list_business_id.append(line[3])
    
# In[]:
"""
--- separate list_business_id into several small lists to break the workload ---

"""
count = 0
length = 5

createVar['list_business_id_'+str(count+1)] = []

for business_id in list_business_id:
    createVar['list_business_id_'+str(count+1)].append(business_id)
    if 0 <= count+1 < 10:
        length = 5
    elif 10 <= count+1 < 20:
        length = 10
    elif 20 <= count+1 < 35:
        length = 20
    elif 35 <= count+1 < 45:
        length = 30
    elif count+1 >= 45:
        length = 40
    if len(createVar.get('list_business_id_'+str(count+1))) == length:
        count += 1
        createVar['list_business_id_'+str(count+1)] = []

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

# In[]
        
"""
--- read review information for business with more than 500 reviews from business folder 
and extract text features from its review texts ---

--- in one while loop ---

"""
#csv.field_size_limit(sys.maxsize)
csv.field_size_limit(500*1024*1024)

filenames = os.listdir(path_text)

# read initial business number
file = open(os.path.join(path_text, 'business number.txt'))
text = file.read()
# read initial business number
# current business number = previous business number + 1
business_number = int(re.findall(r'\d+', text)[0]) + 1

# In[]
""" 
--- define starting point ---
--- DONT RUN THIS SENTENCE AGAIN IF ANY BUSINESS HAS BEEN FINISHED!!! ---

"""
business_number = 0

# In[]
while business_number <= len(list_business_id)-1:
    business_id = list_business_id[business_number]
    if str(business_id)+'.json' in filenames:
        business_number += 1
        continue
    
    print('\n','\n','\n','\n')
    print('current business is: '+str(business_id),'\n','starting time is: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    path_business = os.path.join(path, 'business', 'food&restaurants', str(business_id)+'.json')
    file = open(path_business,'r',encoding='utf-8')
    dict_business = {}
    for line in file.readlines():
        dict_business = json.loads(line)
    print('number of reviews:', str(len(dict_business['Reviews'])))
        
    # preprocessing
    preprocessing(dict_business,stemmer,list_stopwords)
    print('\n',str(business_id), ': preprocessing finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))        
    
    # calculate total term frequency
    dict_wordcount = {}
    total_term_frequency(dict_wordcount, dict_business)
    print('\n',str(business_id), ': total term frequency finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print('total num of words:', str(len(dict_wordcount)))
    
    # calculate document frequency
    document_frequency(dict_wordcount, dict_business)
    print('\n',str(business_id), ': document frequency finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # calculate inverse document frequency
    inverse_document_frequency(dict_wordcount, dict_business)
    print('\n',str(business_id), ': inverse document frequency finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # remove the top 50 and DF<25 unigrams regarding DF
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
    print('length of control list:', str(len(list_wordcount_DF)))
    
    # construct dictionary for controlled vocabulary
    dict_ctrlvoc_TF = {}
    # count term frequency for each document
    for review_id in dict_business['Reviews'].keys():
        dict_ctrlvoc_TF[review_id] = {}
        for word in list_wordcount_DF:
            dict_ctrlvoc_TF[review_id][word[0]] = dict_business['Reviews'][review_id]['text_preprocessed'].count(word[0])
    
    # convert dictionary into dataframe
    df_ctrlvoc_TF = pd.DataFrame.from_dict(dict_ctrlvoc_TF).T.astype(np.float64)
    
    # dataframe for saving probabilities of term frequencies
    df_ctrlvoc_prob = df_ctrlvoc_TF.astype(np.float64)
    
    # calculate probability for each word
    average_term_probability(df_ctrlvoc_prob,df_ctrlvoc_TF,dict_wordcount)
    print('\n',str(business_id), ': average term probability finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # dataframe for saving average TF-IDF weighting
    df_ctrlvoc_weight = df_ctrlvoc_TF.astype(np.float64)
    
    # calculate average TF-IDF weighting
    # weight(term, document) = TF(term, document) * IDF(term)
    average_weighting(df_ctrlvoc_weight,df_ctrlvoc_TF,dict_wordcount)
    print('\n',str(business_id), ': average TF-IDF weighting finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # save two new variables into dict_business
    for review_id in df_ctrlvoc_TF.mean(axis=1).index:
        # save mean probability of TF/TTF
        dict_business['Reviews'][review_id]['mean_prob'] = df_ctrlvoc_TF.mean(axis=1)[review_id]
        # save weight = TF*IDF
        dict_business['Reviews'][review_id]['mean_weight'] = df_ctrlvoc_weight.mean(axis=1)[review_id]
    
    # sentiment analysis 
    for review_id in dict_business['Reviews'].keys():
        dict_business['Reviews'][review_id]['sentiment'] = TextBlob(dict_business['Reviews'][review_id]['text']).sentiment
        dict_business['Reviews'][review_id]['sentiment_polarity'] = dict_business['Reviews'][review_id]['sentiment'][0]
        dict_business['Reviews'][review_id]['sentiment_subjectivity'] = dict_business['Reviews'][review_id]['sentiment'][1]
    print('\n',str(business_id), ': sentiment analysis finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # save current business with its text features into .json file
    with open(os.path.join(path_text, str(business_id) + '_with text feature' + '.json'), 'w+', encoding="utf-8") as outfile:
        json.dump(dict_business, outfile, ensure_ascii=False)         
    print('\n',str(business_id), ': current business finished')
    print('current time is:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # save current business number
    file = open(os.path.join(path_text, 'business number.txt'),'r+')
    file.write('business number '+str(business_number)+' has been finished')
    file.close()   
    
    # next business number
    business_number += 1
    
        
        

        
        



