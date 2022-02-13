# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:17:00 2020

@author: Yichen Jiang
"""

"""
--- this file is for checking and comparing the results of combined dataset ---

"""
# In[]
import csv
import os
import pandas as pd
from tqdm import tqdm
import json
import ast

# In[]
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

path_coef = os.path.join(path,'yelp_filtered set','coefficients')

filenames_coef = os.listdir(path_coef)


# import list of businesses (with over 100 reviews/year)
df_list = pd.read_csv(os.path.join(path,'yelp_filtered set','business with 100 reviews per year in average and non-zero coef_lasso.csv'),index_col = 0)
list_business = list(df_list.index.values)

# dict matching business_id and filename
dict_match = {}
for business_id in tqdm(list_business):
    for filename in filenames_coef:
        if business_id in filename:
            dict_match[business_id] = filename
            
# In[]
"""
--- import coefs ---

"""
dict_business = {}
for business_id in tqdm(list_business):

    dict_business[business_id] = {}

    # lasso(original)
    with open(os.path.join(path_coef,dict_match[business_id]),'r', encoding = 'utf-8') as file:
        df_temp = pd.read_csv(file,header=1,index_col=0)
        file.close()
        for index in df_temp.index:
            if 'decay' in index: # get coefficients
                dict_business[business_id][index] = df_temp.loc[index]['0']

# In[]
dict_count = {}
for filename in tqdm(list_business):
    if filename not in dict_count.keys():
        dict_count[filename] = {}
    for variable in dict_business[filename].keys():
        if variable.strip('_'+variable.split('_')[len(variable.split('_'))-1]) not in dict_count[filename].keys():
            dict_count[filename][variable.strip('_'+variable.split('_')[len(variable.split('_'))-1])] = 0
        if dict_business[filename][variable] != 0.0:
            dict_count[filename][variable.strip('_'+variable.split('_')[len(variable.split('_'))-1])] = 1
            # (here I only count once for all decay values for one variable)
            # e.g. decay_1star_1 and decay_1star_0.05 are all significant, then just count 1star = 1, only once
# In[]

df_count = pd.DataFrame.from_dict(dict_count).T  
# In[]
"""
--- stats ---

"""

list_sum = []
list_percentage = []
for column in df_count.columns:
    list_sum.append(df_count[column].sum())
    list_percentage.append(df_count[column].sum()/len(df_count))

df_stats= pd.DataFrame({'count':list_sum,'percentage':list_percentage},index=df_count.columns)


# In[]
print(df_stats['count'].sum())

df_stats = df_stats.sort_values(by=['count'],ascending = False)

# In[]
"""
--- export df_stats --- 

"""
df_stats.to_csv(os.path.join(path,'yelp_filtered set','significance count_originalselected_lasso.csv'), index=True, quoting=1)


# In[]
"""
---- check coef significance in all original business data ---

"""
"""
--- import coefs ---

"""

dict_business = {}
for filename in tqdm(filenames_coef):
    business_id = filename.split('_time')[0]
    dict_business[business_id] = {}

    # lasso(original)
    with open(os.path.join(path_coef,filename),'r', encoding = 'utf-8') as file:
        df_temp = pd.read_csv(file,header=1,index_col=0)
        file.close()
        for index in df_temp.index:
            if 'decay' in index: # get coefficients
                dict_business[business_id][index] = df_temp.loc[index]['0']

# In[]
dict_count = {}
for filename in filenames_coef:
    business_id = filename.split('_time')[0]
    if filename not in dict_count.keys():
        dict_count[business_id] = {}
    for variable in dict_business[business_id].keys():
        if variable.strip('_'+variable.split('_')[len(variable.split('_'))-1]) not in dict_count[business_id].keys():
            dict_count[business_id][variable.strip('_'+variable.split('_')[len(variable.split('_'))-1])] = 0
        if dict_business[business_id][variable] != 0.0:
            dict_count[business_id][variable.strip('_'+variable.split('_')[len(variable.split('_'))-1])] = 1
            # (here I only count once for all decay values for one variable)
            # e.g. decay_1star_1 and decay_1star_0.05 are all significant, then just count 1star = 1, only once

# In[]
df_count = pd.DataFrame.from_dict(dict_count).T  
# In[]
"""
--- stats ---

"""

list_sum = []
list_percentage = []
for column in df_count.columns:
    list_sum.append(df_count[column].sum())
    list_percentage.append(df_count[column].sum()/len(df_count))

df_stats= pd.DataFrame({'count':list_sum,'percentage':list_percentage},index=df_count.columns)


# In[]
print(df_stats['count'].sum())


df_stats = df_stats.sort_values(by=['count'],ascending = False)

# In[]
"""
--- export df_stats --- 

"""
df_stats.to_csv(os.path.join(path,'yelp_filtered set','significance count_original_lasso.csv'), index=True, quoting=1)

