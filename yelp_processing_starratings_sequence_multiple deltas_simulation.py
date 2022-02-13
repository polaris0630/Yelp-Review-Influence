# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:26:22 2020

@author: Yichen Jiang
"""
"""
--- this file is for generating variables for simulated data ---

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
import random
from sklearn.utils import shuffle  

# In[]:
"""
--- import business data ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

path_business = os.path.join(path,'yelp_filtered set','businesses')

filenames_business = os.listdir(path_business)

# In[]:
"""
--- import simulation data ---

"""
path_simulation = os.path.join(path,'yelp_filtered set','simulations')

filenames_simulation = os.listdir(path_simulation)

# In[]
"""
--- import business with 100 reviews/year in average ---

"""
df_list = pd.read_csv(os.path.join(path,'yelp_filtered set','business with 100 reviews per year in average and non-zero coef_lasso extra.csv'),index_col = 0)

list_business = list(df_list.index.values)


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


# create lookup dictionary and table
#df_decaytable, dict_decaytable = create_decay_table(list_deltas)

# create dict_calendar for time-checking
dict_calendar = create_calendar()

# In[]
dict_check = {}
for filename in filenames_business:
    business_id = filename[0:len(filename)-10]
    dict_check[business_id] = filename


filenames_data = os.listdir(os.path.join(path,'yelp_filtered set','simulations_dataframes'))
knot_base = 'time'

# In[]
"""
--- main process ---

"""

for business_id in tqdm(list_business):
    
    if 'df_'+str(business_id)+'_time_simulation_100.csv' in filenames_data:
        continue
    
    print('\n')
    print('current time is:', datetime.now(), 'business_id is:', str(business_id))
    
    
    """ --- import simulation data --- """
    df_simulation = pd.read_csv(os.path.join(path_simulation,'df_'+business_id+'_time_simulation_multinomial.csv'))
    
    """ create dict_business for saving business information and simulated data(star rating)"""
    dict_business = {}
    
    """ import dict_business """
    filename = dict_check[business_id]
    with open(os.path.join(path_business,filename),'r',encoding='utf-8') as file:
        for line in file.readlines():
            dict_business = json.loads(line)
    
    
    """ str -> float or datetime"""
    """ eliminate duplicate features & combine similar features """
    list_sent = ['mean_prob','sentiment_polarity','sentiment_subjectivity']
    for review_id in dict_business['Reviews'].keys():
        # convert to datatime
        dict_business['Reviews'][review_id]['date'] = datetime.strptime(dict_business['Reviews'][review_id]['date'], "%Y-%m-%d %H:%M:%S")
        # aggregate votes
        dict_business['Reviews'][review_id]['votes'] = dict_business['Reviews'][review_id]['cool']+dict_business['Reviews'][review_id]['useful']+dict_business['Reviews'][review_id]['funny']
        del dict_business['Reviews'][review_id]['cool']
        del dict_business['Reviews'][review_id]['useful']
        del dict_business['Reviews'][review_id]['funny']
        # remove duplicate features
        del dict_business['Reviews'][review_id]['friend_count']
        del dict_business['Reviews'][review_id]['text_length']
        del dict_business['Reviews'][review_id]['textclean_length']
        del dict_business['Reviews'][review_id]['mean_weight']
        # str -> float
        for variable in dict_business['Reviews'][review_id].keys():
            if 'star' in variable:
                continue
            elif variable in list_variables:
                dict_business['Reviews'][review_id][variable] = float(dict_business['Reviews'][review_id][variable])
        # add interactions as variables into dict_business
        for sent in list_sent:
            dict_business['Reviews'][review_id]['stars'+'*'+str(sent)] = float(dict_business['Reviews'][review_id]['stars'])*dict_business['Reviews'][review_id][sent]
    
    
    """ run all simulation data (100) and obtain result for lasso regression on simulation """
    for num in tqdm(range(1,101)):  
        # list of simulated star ratings
        list_stars = list(df_simulation[str(num)].copy())
        
        """ dictionary -> dataframe """
        df_business = pd.DataFrame.from_dict(dict_business['Reviews']).T # one-hot encoding will be performed later when simulated star ratings have been added
        """ sort by datetime """
        df_business = df_business.sort_values(by = 'date', axis = 0, ascending = True)
        list_dates = list(df_business['date'].copy())
        
        # shuffle the reviews and corresponds to each simualted star ratings
        df_business = shuffle(df_business)
        # replace review time
        df_business['date'] = list_dates
        
        """ add average_stars"""
        add_average_star(df_business)
        
        """ overwrite star ratings with new star ratings """
        df_business['stars'] = list_stars

        """ one-hot encoding for new (simulated) star ratings """
        df_business = one_hot_star_df(df_business)

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
        df_data.to_csv(os.path.join(path,'yelp_filtered set','simulations_dataframes', 'df_'+str(business_id)+'_'+str(knot_base)+'_simulation_'+str(num)+'.csv'), index=False, quoting=1)

# In[]
""" save the current position """

# export business_id
file = open(os.path.join(path,'simulation_dataframe', 'business and base.txt'),'r+')
file.truncate()
file.write(str(business)+','+str(num)+'\n')
file.close()  




















# In[]
"""
--- backup code ---

"""

# In[]

index_business = list_business.index(business)
length = len(list_business)
list_business = list_business[index_business:length-1]












# In[]
count = 0
# clear dict_business for new information
dict_business = {}
dict_business['RestaurantInfo'] = dict_business_raw['RestaurantInfo']
dict_business['Reviews'] = {}
for review_id in dict_business_raw['Reviews'].keys():
    if review_id not in dict_business['Reviews'].keys():
        dict_business['Reviews'][review_id] = dict_business_raw['Reviews'][review_id]
    # revise star raring with shuffled simulated data
    dict_business['Reviews'][review_id]['stars'] = df_temp[count]
    count += 1







        
""" calculate decay values"""
# dictionary for saving effective events (events with decay values > 1e-7)
dict_effective = {}
# dictionary for saving decay values for each event
dict_decayvalues = {}

calculate_decay_values(df_business, list_variables, list_deltas, dict_decaytable, dict_effective, dict_decayvalues)

""" get dataframe """
# get dataframe of independent variables and save it into .csv for running lasso model in R
df_variables = pd.DataFrame(dict_decayvalues).T.drop('order',axis=1)
    
