# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:40:00 2019

@author: Yichen Jiang
"""

"""
--- this file is for checking yelp business analysis based on time and order ---

"""

# In[]
import csv
import os
import pandas as pd
from tqdm import tqdm

# In[]
# initialization
dict_business = {}

# In[]
"""
--- import data ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

filenames_coef = os.listdir(os.path.join(path,'results'))

filenames_data = os.listdir(os.path.join(path,'dataframe_ready_to_use'))

# import list of businesses (with over 100 reviews/year)
df_list = pd.read_csv(os.path.join(path,'business with 100 reviews per year in average.csv'),index_col = 0)

list_business = list(df_list.index.values)

# In[]

"""
--- read coefficient files and preprocess ---

"""
# save business and knot_base into dictionary

count_year = 0
zero_coefs = True 
for filename in tqdm(filenames_coef):
    if filename.endswith('.csv') and filename.split('_coefficients')[0] in list_business and filename.split('_coefficients')[0] not in dict_business.keys():
        dict_business[filename.split('_coefficients')[0]] = {}
        with open(os.path.join(path,'results',filename),'r', encoding = 'utf-8') as file:
            df_temp = pd.read_csv(file,header=1,index_col=0)
            file.close()
            for index in df_temp.index:
                if 'decay' in index: # get coefficients
                    dict_business[filename.split('_coefficients')[0]]['coef_'+index] = df_temp.loc[index]['0']
                    # check if all coefs are zeros
                    if df_temp.loc[index]['0'] > 0.0:
                        zero_coefs = False
                if 'spline' in index: # get number of years
                    count_year += 1
            dict_business[filename.split('_coefficients')[0]]['year_range'] = count_year-1
            count_year = 0
        # get number of reviews
        with open(os.path.join(path,'dataframe_ready_to_use','df_'+filename.split('_coefficients')[0]+'.csv')) as file:
            df_temp = pd.read_csv(file,header=0)
            file.close()
            dict_business[filename.split('_coefficients')[0]]['num_of_reviews'] = len(df_temp)
        # check if all coefs are zero
        dict_business[filename.split('_coefficients')[0]]['zero_coefs'] = zero_coefs
        zero_coefs = True 
        # knot_base
        dict_business[filename.split('_coefficients')[0]]['knot_base'] = filename.split('_coefficients')[0].split('_')[len(filename.split('_coefficients')[0].split('_'))-1]
            
# In[]
"""
--- convert dict -> df ---

"""
df_business = pd.DataFrame.from_dict(dict_business).T

# In[]
"""
--- counts: significant coefficients ---

"""
list_variables = ['1star',
 '2star',
 '3star',
 '4star',
 '5star',
 'cool',
 'funny',
 'useful',
 'average_stars',
 'friend_count',
 'elite_count',
 'review_count',
 'fan_count',
 'yelping_since',
 'mean_prob',
 'mean_weight',
 'sentiment_polarity',
 'sentiment_subjectivity',
 'text_length',
 'textclean_length']

dict_count = {}

for business in dict_business:
    if business not in dict_count.keys():
        dict_count[business] = {}
    for variable in list_variables:
        if variable not in dict_count[business].keys():
            dict_count[business][variable] = 0
        for coef in dict_business[business].keys():
            if variable in coef and dict_business[business][coef] != 0.0:
                dict_count[business][variable] = 1
                
df_count = pd.DataFrame.from_dict(dict_count).T            


# In[]
"""
--- statistics ---

"""
list_sum = []
list_percentage = []
for column in df_count.columns:
    list_sum.append(df_count[column].sum())
    list_percentage.append(df_count[column].sum()/len(df_count))

df_stats = pd.DataFrame({'count':list_sum,'percentage':list_percentage},index=df_count.columns)

df_stats.sort_values(['percentage'],ascending=0,inplace=True)


