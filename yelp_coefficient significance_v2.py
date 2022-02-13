# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:10:01 2020

@author: Yichen Jiang
"""

"""
--- this file is for checking coefficient significance of simulated data on both datasets ---

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
import copy

# In[]
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'
path_original = os.path.join(path,'yelp_filtered set','coefficients')
path_simulation = os.path.join(path,'yelp_filtered set','simulations_coefficients')

filenames_original = os.listdir(path_original)
filenames_simulation = os.listdir(path_simulation)

# In[]
"""
--- list_business: businesses in the simulation set ---
--- dict for matching business_id and original coef filename in filenames_original ---

"""

list_business = []
for filename in tqdm(filenames_simulation):
    business_id = filename.split('_simulation')[0]
    if business_id not in list_business:
        list_business.append(business_id)
        
dict_match = {}
for filename in tqdm(filenames_original):
    business_id = filename.split('_time')[0]
    if business_id in list_business:
        dict_match[business_id] = filename

# In[]
"""
--- import coef and calculate coef significance ---

"""
dict_sig = {}
knot_base = 'time'

for business_id in tqdm(list_business):


    """ --- import coefficients from both simulation and original model --- """
    """ --- original --- """
    filename = dict_match[business_id]
    df_coef = pd.read_csv(os.path.join(path_original,filename),header=1,index_col=0)
    # remove coefficients of b-spline
    index_bspline = list(df_coef.index.values).index('b_spline_0')
    df_coef = df_coef.iloc[:index_bspline,:]

    """ --- simulation data --- """
    for count in range(1,101):
        filename = str(business_id)+'_simulation_'+str(count)+str('_time_coefficients.csv')
        df_temp = pd.read_csv(os.path.join(path_simulation,filename),header=1,index_col=0)
        # remove coefficients of b-spline
        df_temp = df_temp.iloc[:index_bspline,:]
        df_coef = pd.concat([df_coef,df_temp],axis=1)
        
    # revise column names
    df_coef.columns = ['real']+list(range(0,count))

    """ --- calculate test statistics --- """
    dict_sig[business_id] = {}
    dict_sig[business_id]['run_num'] = count
    
    for variable in df_coef.index:
        dict_sig[business_id][variable] = {}
        real_coef = abs(df_coef.loc[variable,'real'])
        count = 0
        for column in df_coef.columns:
            if column == 'real':
                continue
            else:
                coef = abs(df_coef.loc[variable,column])
                if coef >= abs(real_coef):#real_coef:
                    count += 1
        dict_sig[business_id][variable]['count'] = count
        dict_sig[business_id][variable]['p-value'] = \
        dict_sig[business_id][variable]['count']/dict_sig[business_id]['run_num']

# In[]
count = 0
for business_id in dict_sig.keys():
    df_temp = pd.DataFrame.from_dict(dict_sig[business_id]).T
    df_temp = df_temp.iloc[1:,:]
    if count == 0:
        df_sig = df_temp.copy()
    else:
        df_sig = pd.concat([df_sig,df_temp],axis=1)
    count +=1

list_temp = []
for business_id in dict_sig.keys():
    list_temp.append(business_id+'_count')
    list_temp.append(business_id+'_p-value')
    
df_sig.columns = list_temp        
        
# In[]
for business_id in dict_sig.keys():
    df_temp = df_sig[business_id+'_p-value'].copy()
    print(df_temp.value_counts())

# In[]
"""
--- combine variables regardless of decay parameter ---

"""
dict_coef = {}

for business_id in dict_sig.keys():
    # save the result in dict_coef for current business
    if business_id not in dict_coef.keys():
        dict_coef[business_id] = {}
    for key in dict_sig[business_id].keys():
        if key == 'run_num':
            continue
        else:
            variable = key.rstrip('_'+key.split('_')[len(key.split('_'))-1])
            # save the smallest p-value this variable has
            if variable not in dict_coef[business_id].keys():
                dict_coef[business_id][variable] = dict_sig[business_id][key]['p-value']
            elif dict_sig[business_id][key]['p-value'] < dict_coef[business_id][variable]:
                dict_coef[business_id][variable] = dict_sig[business_id][key]['p-value']

df_coef = pd.DataFrame.from_dict(dict_coef).T

# In[]
"""
--- count number of significant coefs all businesses have ---

"""

dict_count = {}

for business_id in dict_coef.keys():
    for variable in dict_coef[business_id].keys():
        if variable not in dict_count.keys():
            dict_count[variable] = {'count':0}
        if dict_coef[business_id][variable] <= 0.05:
            dict_count[variable]['count'] += 1

df_count = pd.DataFrame.from_dict(dict_count).T
df_count['percentage'] = df_count['count']/len(dict_coef)

# In[]
plt.figure(figsize=(20,15))
plt.hist(df_coef['decay_5star'],bins=50)     
        
# In[]
""" --- sort & export --- """

df_count = df_count.sort_values(by=['count'],ascending = False)

df_count.to_csv(os.path.join(path,'yelp_filtered set','significance count_simulaiton_lasso.csv'), index=True, quoting=1)
        
        
        
