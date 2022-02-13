# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 00:31:57 2020

@author: Yichen Jiang
"""

"""
--- this file is for running Hawkes processes and creating features based on the changes of combining variables,
removing duplicate variables ---

"""

# In[]
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import bspline
import bspline.splinelab as splinelab
import sys
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'
sys.path.append(path)
from yelp_functions import *
from scipy.interpolate import splrep, BSpline

# In[]:
"""
--- import business data ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

path_business = os.path.join(path,'yelp_filtered set','businesses')

filenames_business = os.listdir(path_business)

filenames_df = os.listdir(os.path.join(path, 'yelp_filtered set', 'dataframes'))

# In[]
"""
--- import business with 100 reviews/year in average ---

"""
df_list = pd.read_csv(os.path.join(path,'business with 100 reviews per year in average.csv'),index_col = 0)

list_business = list(df_list.index.values)

# In[]
"""
--- OR! run all businesses with order-based and time-based b-spline ---

"""
list_business = []

for filename in tqdm(filenames_business):
    business_id = filename.split('_with')[0]
    business = str(business_id)+'_order'
    list_business.append(business)
    business = str(business_id)+'_time'
    list_business.append(business)

# In[]
"""
--- OR! extract business_id from filenames ---

"""
dict_match = {}
for filename in filenames_business:
    if '2020' in filename:
        business_id = filename.split('_2020')[0]
    else:
        business_id = filename.split('_2019')[0]
    dict_match[business_id] = filename
    
# In[]
"""
--- input delta ---

"""
# list of deltas
list_deltas = [0.005, 0.05, 0.1, 1, 5]

# list of variables
list_variables = []
for i in range(0,5):
    list_variables.append(str(i+1)+'star')
list_variables += ['votes', 'average_stars', 'elite_count', 'review_count', 'fan_count', 'yelping_since', 'mean_prob', 'sentiment_polarity', 'sentiment_subjectivity']

list_variables += ['stars*mean_prob','stars*sentiment_polarity','stars*sentiment_subjectivity']

bspline_order = 3

#knot_base = 'time' #'time' or 'order'


# create lookup dictionary and table
# df_decaytable, dict_decaytable = create_decay_table(list_deltas)

# create dict_calendar for time-checking
dict_calendar = create_calendar()

# In[]
"""
--- main process ---

"""
filenames_df = os.listdir(os.path.join(path, 'yelp_filtered set', 'dataframes'))
#filenames_df = os.listdir(os.path.join(path, 'dataframe_interaction'))

for business_id in tqdm(dict_match.keys()):
    """
    print('\n')
    # get business_id and knot_base
    if 'order' in business:
        business_id = business.split('_order')[0]
        knot_base = 'order'
    else:
        business_id = business.split('_time')[0]
        knot_base = 'time'
    """
    knot_base = 'time'
    """
    # if already processed, then pass
    if 'df_'+str(business_id)+'_'+str(knot_base)+'.csv' in filenames_df:
        continue
    """
    print('current business is:', business_id, ', knot_base is:', knot_base)
    print('current time is:', datetime.now())
    
    dict_business = {}
    
    with open(os.path.join(path_business,dict_match[business_id]),'r',encoding='utf-8') as file:
        for line in file.readlines():
            dict_business = json.loads(line)
    
    """ one-hot for star ratings"""
    one_hot_star(dict_business)        
    
    """ str -> float or datetime"""
    """ eliminate duplicate features & combine similar features """
    list_sent = ['mean_prob','sentiment_polarity','sentiment_subjectivity']
    for review_id in dict_business['Reviews'].keys():
        # convert to datatime
        dict_business['Reviews'][review_id]['date'] = datetime.strptime(dict_business['Reviews'][review_id]['date'], "%Y-%m-%d %H:%M:%S")
        # aggregate votes
        dict_business['Reviews'][review_id]['votes'] = abs(int(dict_business['Reviews'][review_id]['cool']))+abs(int(dict_business['Reviews'][review_id]['useful']))+abs(int(dict_business['Reviews'][review_id]['funny']))
        del dict_business['Reviews'][review_id]['cool']
        del dict_business['Reviews'][review_id]['useful']
        del dict_business['Reviews'][review_id]['funny']
        # remove duplicate features
        del dict_business['Reviews'][review_id]['friend_count']
        del dict_business['Reviews'][review_id]['text_length']
        del dict_business['Reviews'][review_id]['textclean_length']
        del dict_business['Reviews'][review_id]['mean_weight']
        # str -> float
        for variable in list_variables:
            if variable == 'average_stars' or '*' in variable:
                continue
            else:
                dict_business['Reviews'][review_id][variable] = float(dict_business['Reviews'][review_id][variable])
        # add interactions as variables into dict_business
        for sent in list_sent:
            dict_business['Reviews'][review_id]['stars'+'*'+str(sent)] = float(dict_business['Reviews'][review_id]['stars'])*dict_business['Reviews'][review_id][sent]
    
    
    """ dictionary -> dataframe"""
    df_business = pd.DataFrame.from_dict(dict_business['Reviews']).T
    """ sort by datetime"""
    df_business = df_business.sort_values(by = 'date', axis = 0, ascending = True)
    
    """ add average_stars"""
    add_average_star(df_business)
    
    
    """ clear dataframe"""
    df_business = pd.concat([df_business['date'],df_business['stars'],df_business[list_variables]],axis=1)
    
    """ get decay array """
    array_decay = create_decay_array(list_deltas,len(df_business))
    
    """ business data -> array """
    array_business = np.array(df_business[list_variables],dtype='float64')
    
    """ result calculation"""
    array_result = np.dot(array_decay, array_business)
    
    """ result array -> df """
    list_columns = []
    
    for i in range(len(list_deltas)):
        if i == 0:
            df_result = pd.DataFrame(array_result[i])
        else:
            df_temp = pd.DataFrame(array_result[i])
            df_result = pd.concat([df_result,df_temp],axis=1)
        for variable in list_variables:
            list_columns.append('decay_'+str(variable)+'_'+str(list_deltas[i]))
            
    df_result.columns = list_columns
    df_result.index = df_business.index
    
    
    """ b spline basis function """
    if knot_base == 'order':
        # get knot vector
        list_knot_order = get_knot_vector(df_business)
        # get b spline basis function
        df_business, df_bspline = b_spline_basis(bspline_order, list_knot_order, df_business)
    elif knot_base == 'time':
        # get knot vector
        list_knot_order, list_knot_time = get_knot_vector_time(df_business)
        # get b spline basis function
        df_business, df_bspline = b_spline_basis_time(bspline_order, list_knot_time, df_business)
    
    # export dataframe
    df_data = pd.concat([df_business['stars'],df_result,df_bspline],axis = 1)
    df_data.to_csv(os.path.join(path, 'yelp_filtered set', 'dataframes' , 'df_'+str(business_id)+'_'+str(knot_base)+'.csv'), index=False, quoting=1)
    # export business_id
    #file = open(os.path.join(path,'dataframe_interaction', 'business_id and knot_base.txt'),'r+')
    #file.truncate()
    #file.write(str(business_id)+','+str(knot_base)+'\n')
    #file.close()   

# In[]            





