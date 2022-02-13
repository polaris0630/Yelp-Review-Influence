# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:33:04 2019

@author: yjian
"""
# In[]:

import csv
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# In[]:

path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

path_data = os.path.join(path,'yelp_dataset', 'yelp_dataset~')

filenames_data = os.listdir(path_data)

# In[]
path_business = os.path.join(path, 'yelp_dataset', 'business')

path_review = os.path.join(path, 'yelp_dataset', 'review')

path_output = os.path.join(path_business, 'food&restaurants')

# In[]:
createVar = locals()

# In[]:
list_restaurants = []
list_columnname = []

# In[]:

filenames = os.listdir(path_business)

for filename in filenames:
    if filename.endswith('.csv'):
        file = open(os.path.join(path_business, filename),'r',encoding='utf-8')
        lines = csv.reader(file)
        
        for line in lines:
            # obtain column name
            if line[0] == '' and list_columnname == []:
                list_columnname.append(line)
            # obtain restaurants
            index_review = list_columnname[0].index('review_count')
            if 'Food' in line[4] or 'Restaurants' in line[4]:
                line[index_review] = int(line[index_review])
                list_restaurants.append(line)

# In[]:
# convert list to dataframe
df_restaurants = pd.DataFrame(columns = list_columnname[0], data = list_restaurants)
# delete old index
del df_restaurants['']
# sort by review_count
df_restaurants = df_restaurants.sort_values(by = 'review_count', axis = 0, ascending = False)
# convert dataframe to csv and save
pd.DataFrame(df_restaurants).to_csv(os.path.join(path_output, 'food&restaurants.csv'), encoding = 'utf-8')

# In[]:
# review_count >= 500 & 1000
df_500 = df_restaurants[df_restaurants.review_count >= 500]
df_1000 = df_restaurants[df_restaurants.review_count >= 1000]
# convert dataframe to csv and save
pd.DataFrame(df_500).to_csv(os.path.join(path_output, 'food&restaurants500.csv'), encoding = 'utf-8')
pd.DataFrame(df_1000).to_csv(os.path.join(path_output, 'food&restaurants1000.csv'), encoding = 'utf-8')

# In[]:
# list with review_count >= 1000

list_500 = df_500.values.tolist()

list_columnname = df_500.columns.values.tolist()
index_business = list_columnname.index('business_id')
# create dictionary for each restaurant
dict_500 = {}

for restaurant in list_500:
    business_id = restaurant[index_business]
    dict_500[business_id] = {'RestaurantInfo':{}, 'Reviews':{}}
    for column in list_columnname:
        index_column = list_columnname.index(column)
        dict_500[business_id]['RestaurantInfo'][column] = restaurant[index_column]

# In[]:
filenames = os.listdir(path_review)

list_reviewcolumn = []

# In[]:
count = 0

for filename in tqdm(filenames):
    count = 0
    if filename.endswith('.csv'):
        file = open(os.path.join(path_review, filename), 'r', encoding = 'utf-8')
        lines = csv.reader(file)
        
        for line in lines:
            
            # obtain column name
            if count == 0:
                list_reviewcolumn = line
                list_reviewcolumn.pop(0)
                # obtain business_id index and review_id index
                index_business = list_reviewcolumn.index('business_id')
                index_review = list_reviewcolumn.index('review_id')
                count += 1
                continue
                
            line.pop(0)    
            business_id = line[index_business]
            # check if this review belongs to restaurants with more than 1000 reviews
            
            if business_id in dict_500.keys():
                
                review_id = line[index_review]
                dict_500[business_id]['Reviews'][review_id] = {}
                
                for i in range(len(line)):
                    column = list_reviewcolumn[i]
                    dict_500[business_id]['Reviews'][review_id][column] = line[i]
                #for column in list_reviewcolumn:
                #    index_column = list_reviewcolumn.index(column)
                #    dict_1000[business_id]['Reviews'][review_id][column] = line[index_column]
                
                #print(dict_500[business_id]['Reviews'][review_id])
            
            
# In[]:
# save dictionaries

for key in dict_500:
    with open(os.path.join(path_output, str(key) + '.json'), 'w+', encoding="utf-8") as outfile:
        json.dump(dict_500[key], outfile, ensure_ascii=False) 

# In[]:
list_500check = []
# check count with review_count
for key in dict_500:
    list_500check.append([key,dict_500[key]['RestaurantInfo']['review_count'], len(dict_500[key]['Reviews'])])
    
    
    
    
    


