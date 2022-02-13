# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:28:15 2019

@author: Yichen Jiang
"""

"""
--- this file is for saving possible functions for yelp_model processing ---

"""
# In[]
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime,timedelta
import pandas as pd
from tqdm import tqdm
import bspline
import bspline.splinelab as splinelab

# In[]:
createVar = locals()

# In[]
def get_date(timestamp):
    date_str = str(timestamp)[0:10]
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return date

def get_date_num(timestamp,initial_timestamp):
    initial_date = get_date(initial_timestamp)
    current_date = get_date(timestamp)
    date_num = (current_date-initial_date).days
    return date_num

# In[]:
"""
--- create dictionary for calendar checking ---

"""
def create_calendar():
    dict_calendar = {}
    for i in range(1990, 2020):
        dict_calendar[str(i)] = {} 
    
    for key in dict_calendar:
        for i in range(0,12):
            if str(i+1) not in dict_calendar[key].keys():
                if int(key) % 4 == 0 and i == 1:
                    dict_calendar[key][str(i+1)] = 29
                elif str(i+1) in ['1', '3', '5', '7', '8', '10', '12']:
                    dict_calendar[key][str(i+1)] = 31
                elif str(i+1) in ['4', '6', '9', '11']:
                    dict_calendar[key][str(i+1)] = 30
                elif str(i+1) == '2':
                    dict_calendar[key][str(i+1)] = 28
    return dict_calendar

# In[]
"""
--- create function for calculating date_time difference ---

"""
def get_time_diff(curr_date,past_date):
    # date should be in datetime format
    time_diff = (curr_date - past_date)
    time_num = time_diff.days + time_diff.seconds/86400
    return time_num

# In[]:
"""
--- define function for creating lookup table for decay values ---

"""

def create_decay_table(list_deltas):
    # set maximum difference of order
    max_orderdiff = 25000 
    # indicator for checking if all decay values are 0
    count = 0
    # create dict_decaytable
    dict_decaytable = {}
    for delta in list_deltas:
        dict_decaytable[delta] = {}
    # order difference
    orderdiff = 0
    while orderdiff <= max_orderdiff:
        for delta in dict_decaytable.keys():
            decay = math.e ** (-(orderdiff) * delta)
            decay = decay * delta
            if decay <= 1e-4:
                decay = 0.0
            dict_decaytable[delta][orderdiff] = decay
            # check if decay value == 0
            if decay == 0.0:
                count += 1
        # if all values == 0
        if count == 5:
            break
        else: count = 0
        orderdiff += 1
    # create dataframe
    df_decaytable = pd.DataFrame(dict_decaytable)
    return df_decaytable, dict_decaytable

# In[]
"""
--- define function for creating decay arrays ---

"""
def create_decay_array(list_deltas,length):
    array = np.zeros((len(list_deltas),length,length))
    
    for i in range(0,length):
        for j in range(i,length):
            orderdiff = j-i
            if orderdiff == 0:
                continue
            else:
                for k in range(len(list_deltas)):
                    decay = math.e ** (-(orderdiff) * list_deltas[k])
                    decay = decay * list_deltas[k]
                    array[k,j,i] = decay

    return array

# In[]
"""
--- define function for extracting user features ---

"""
def get_user_features(business_id, path, dict_business):
    # read user file
    dict_business_user = {}
    with open(os.path.join(path, 'business', 'business with more than 500 reviews', str(business_id) + '_with user information' + '.json'), 'r', encoding="utf-8") as file:
        for line in file.readlines():
            dict_business_user = json.loads(line)
    # extract features
    for review_id in dict_business_user['Reviews'].keys():
        friend_count = dict_business_user['Reviews'][review_id]['user_info']['friends'].count(',')
        elite_count = dict_business_user['Reviews'][review_id]['user_info']['elite'].count(',')
        review_count = dict_business_user['Reviews'][review_id]['user_info']['review_count']
        fan_count = dict_business_user['Reviews'][review_id]['user_info']['fans']
        yelping_since = datetime.strptime(dict_business_user['Reviews'][review_id]['user_info']['yelping_since'], "%Y-%m-%d %H:%M:%S")
        
        if friend_count != 0:
            friend_count += 1
        
        if elite_count != 0:
            elite_count += 1
        
        review_time = datetime.strptime(dict_business_user['Reviews'][review_id]['date'], "%Y-%m-%d %H:%M:%S")
        yelping_since = get_time_diff(review_time,yelping_since)
        
        dict_business['Reviews'][review_id]['friend_count'] = friend_count
        dict_business['Reviews'][review_id]['elite_count'] = elite_count
        dict_business['Reviews'][review_id]['review_count'] = review_count
        dict_business['Reviews'][review_id]['fan_count'] = fan_count
        dict_business['Reviews'][review_id]['yelping_since'] = yelping_since

# In[]
"""
--- define function for extracting text features ---

"""
def get_text_features(business_id, path, dict_business):
    # read text file
    dict_business_text = {}
    with open(os.path.join(path, 'business with text features', str(business_id) + '_with text feature' + '.json'),'r',encoding='utf-8') as file:
        for line in file.readlines():
            dict_business_text = json.loads(line)
    # extract features
    for review_id in dict_business['Reviews'].keys():
        dict_business['Reviews'][review_id]['mean_prob'] = dict_business_text['Reviews'][review_id]['mean_prob']
        dict_business['Reviews'][review_id]['mean_weight'] = dict_business_text['Reviews'][review_id]['mean_weight']
        dict_business['Reviews'][review_id]['sentiment_polarity'] = dict_business_text['Reviews'][review_id]['sentiment_polarity']
        dict_business['Reviews'][review_id]['sentiment_subjectivity'] = dict_business_text['Reviews'][review_id]['sentiment_subjectivity']
        dict_business['Reviews'][review_id]['text_length'] = dict_business_text['Reviews'][review_id]['text_length']
        dict_business['Reviews'][review_id]['textclean_length'] = dict_business_text['Reviews'][review_id]['textclean_length']

# In[]
"""
--- define function for one-hot encoding for star ratings ---

"""
def one_hot_star(dict_business):
    for review_id in dict_business['Reviews'].keys():
        star = int(float(dict_business['Reviews'][review_id]['stars']))
        for i in range(0,5):
            if star == i+1:
                dict_business['Reviews'][review_id][str(i+1)+'star'] = 1
            else:
                dict_business['Reviews'][review_id][str(i+1)+'star'] = 0
                
# In[]
"""
--- define function for one-hot encoding for star ratings in dataframe---

"""
def one_hot_star_df(df_business):
    dict_stars = {}
    for index in df_business.index:
        star = int(df_business.loc[index]['stars'])
        if index not in dict_stars.keys():
            dict_stars[index] = {}
            for i in range(1,6):
                if i == star:
                    dict_stars[index][str(i)+'star'] = 1
                else:
                    dict_stars[index][str(i)+'star'] = 0
    df_stars = pd.DataFrame.from_dict(dict_stars).T

    df_business = pd.concat([df_business,df_stars],axis=1)
    return df_business

# In[]
"""
--- define function for adding average_star as a variable ---

"""
def add_average_star(df_business):
    cumulative_stars = 0.0
    average_stars = 0.0
    list_avgstars = []
    
    for i in range(len(df_business.index)):
        cumulative_stars += float(df_business.iloc[i]['stars'])
        average_stars = cumulative_stars/(i + 1)
        list_avgstars.append(average_stars)    

    df_business['average_stars'] = list_avgstars
    
# In[]
"""
--- define main function ---

"""
def calculate_decay_values(df_business, list_variables, list_deltas, dict_decaytable,dict_effective,dict_decayvalues):

    # list for saving uninfluential events
    list_temp = []
    
    for i in tqdm(range(len(df_business.index))):
        review_id = df_business.index[i]
        dict_decayvalues[review_id] = {'order': i}
        for variable in list_variables:
            if variable not in dict_effective.keys():
                dict_effective[variable] = {}
            for delta in list_deltas:
                if delta not in dict_effective[variable].keys():
                    dict_effective[variable][delta] = {}
                # save current event into dict_effective
                dict_effective[variable][delta][i] = {'coef': float(df_business.iloc[i][variable])}
                
                # calculate and update decay values of all effective events
                for event_num in dict_effective[variable][delta].keys():
                    coef = dict_effective[variable][delta][event_num]['coef']
                    order_diff = i - event_num
                    # check decaytable and obtain corresponding decay value
                    decay = coef * dict_decaytable[delta][order_diff]
                    dict_effective[variable][delta][event_num]['decay'] = decay
                
                # calculate cumulative decay values and save it into dict_decayvalues
                decay = 0.0  # initialize decay
                for event_num in dict_effective[variable][delta].keys():
                    # current event couldn't have influence immediately
                    if event_num == i:
                        continue
                    decay += dict_effective[variable][delta][event_num]['decay']
                dict_decayvalues[review_id]['decay_'+str(variable)+'_'+str(delta)] = decay
                
                # remove uninfluential events
                # save uninfluential events
                for event_num in dict_effective[variable][delta].keys():
                    if dict_effective[variable][delta][event_num]['decay'] <= 1e-7:
                        list_temp.append(event_num)
                # remove uninfluential events
                for event_num in list_temp:
                    dict_effective[variable][delta].pop(event_num)
                list_temp = [] # clear list

# In[]
"""
--- create function for creating b spline basis elements based on order ---

"""
def get_knot_vector(df_business):

    list_knot_order = [0] # startpoint
    start_year = df_business['date'][0].year # start year
    # get array of years
    np_year=np.array([df_business['date'][i].year-start_year for i in range(len(df_business.index))]).astype('float64')
    year = 0
    for i in range(len(np_year)):
        if np_year[i] > year:
            list_knot_order.append(i-0.5)
            year = np_year[i]
    list_knot_order.append(len(df_business)-1)
    return list_knot_order

# In[]
"""
--- create function for creating b spline basis elements based on time (date)---

"""
def get_knot_vector_time(df_business):
    list_knot_order = [0] # startpoint (by order)
    list_knot_time = [0.0] # startpoint (by time)
    # get start_date
    start_date = df_business['date'][0]
    # get arrary of years
    np_year=np.array([df_business['date'][i].year-start_date.year for i in range(len(df_business.index))]).astype('float64')
    year = 0
    for i in range(len(np_year)):
        if np_year[i] > year:
            list_knot_order.append(i-0.5)
            list_knot_time.append(get_time_diff(df_business['date'][i],start_date))
            year = np_year[i]
    list_knot_order.append(len(df_business)-1)
    list_knot_time.append(get_time_diff(df_business['date'][len(df_business)-1],start_date))
    return list_knot_order, list_knot_time

# In[]
"""
--- define function for b spline basis function based on order ---

"""
def b_spline_basis(order, knot_vector_order, df_business):

    p = order              # order of spline (as-is; 3 = cubic)
    #nknots = 12       # number of knots to generate (here endpoints count only once)
    tau = np.arange(len(df_business))  # collocation sites (i.e. where to evaluate)

    k = splinelab.augknt(knot_vector_order, p)  # add endpoint repeats as appropriate for spline order p
    B = bspline.Bspline(k, p)       # create spline basis of order p on knots k
    
    df_bspline = pd.DataFrame(B.collmat(tau))    # collocation matrix for function value at sites tau
    df_bspline.index = df_business.index
    # rename column
    dict_column = {}
    for column in df_bspline.columns:
        dict_column[column] = 'b_spline_'+str(column)
    df_bspline = df_bspline.rename(columns=dict_column)
    
    return df_business, df_bspline

# In[]
"""
--- define function for bspline basis function based on time (date) ---

"""
def b_spline_basis_time(order, knot_vector_time, df_business):
    
    p = order              # order of spline (as-is; 3 = cubic)
    # convert datetime -> time number
    list_time = [] 
    # get start_date of current business
    start_date = df_business['date'][0]
    for i in range(len(df_business)):
        time_num = get_time_diff(df_business['date'][i],start_date)
        list_time.append(time_num)
    tau = np.asarray(list_time)  # collocation sites (i.e. where to evaluate)
    
    k = splinelab.augknt(knot_vector_time, p)  # add endpoint repeats as appropriate for spline order p
    B = bspline.Bspline(k, p)       # create spline basis of order p on knots k
    
    df_bspline = pd.DataFrame(B.collmat(tau))    # collocation matrix for function value at sites tau
    df_bspline.index = df_business.index
    # rename column
    dict_column = {}
    for column in df_bspline.columns:
        dict_column[column] = 'b_spline_'+str(column)
    df_bspline = df_bspline.rename(columns=dict_column)
        
    return df_business, df_bspline


