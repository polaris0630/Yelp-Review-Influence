# -*- coding: utf-8 -*-
"""
Created on Sat May  2 02:35:39 2020

@author: Yichen Jiang
"""

"""
--- this file is for computing standardized coefs for original observed coefs && simulation coefs 
    obtained from lasso model ---

--- 1. coef * sd. ---
--- 2. coef / sd. ---

"""

# In[]
import csv
import os
import pandas as pd
from tqdm import tqdm
import json
import ast
import math

# In[]
""" --- import data --- """
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'
path_df = os.path.join(path,'yelp_filtered set','dataframes')
path_sim_df = os.path.join(path,'yelp_filtered set','simulations_dataframes')
path_coef = os.path.join(path,'yelp_filtered set','coefficients')
path_coef_multiply = os.path.join(path,'yelp_filtered set','coefficients multiply std')
path_coef_divide = os.path.join(path,'yelp_filtered set','coefficients divide std')
path_sim_coef = os.path.join(path,'yelp_filtered set','simulations_coefficients')
path_sim_coef_multiply = os.path.join(path,'yelp_filtered set','simulations_coefficients multiply std')
path_sim_coef_divide = os.path.join(path,'yelp_filtered set','simulations_coefficients divide std')


filenames_df = os.listdir(path_df)
filenames_sim_df = os.listdir(path_sim_df)
filenames_coef = os.listdir(path_coef)
filenames_sim_coef = os.listdir(path_sim_coef)

list_business = list(pd.read_csv(os.path.join(path,'yelp_filtered set','business with 100 reviews per year in average and non-zero coef_lasso.csv'), index_col = 0).index)

# In[]
""" --- standardize observed coef by multiplying or dividing std. of each variable --- """

for filename in tqdm(filenames_coef):
    business_id = filename.split('_time')[0]
    # import business dataframe
    with open(os.path.join(path_df,'df_'+str(business_id)+'_time.csv'),'r', encoding = 'utf-8') as file:
        df_business = pd.read_csv(file,header=0)
        file.close()
    df_std = df_business.describe().loc['std']
    # import coef 
    with open(os.path.join(path_coef,filename),'r', encoding = 'utf-8') as file:
        df_coef = pd.read_csv(file,header=1,index_col=0)
        file.close()
    # coef multiply std && coef divide std
    df_multiply = pd.DataFrame()
    df_divide = pd.DataFrame()
    for index in df_coef.index.values:
        df_multiply = df_multiply.append(df_coef.loc[index])
        df_divide = df_divide.append(df_coef.loc[index])
        # if non-zero coef
        if 'decay' in index and df_coef.loc[index]['0'] > 0:
            if 'stars.' in index:
                index_std = index[0:len(index.split('.')[0])]+'*'+index[len(index.split('.')[0])+1:len(index)]
                df_multiply.loc[index]['0'] = df_coef.loc[index]['0']*df_std.loc[index_std]
                df_divide.loc[index]['0'] = df_coef.loc[index]['0']/df_std.loc[index_std]
            else:
                df_multiply.loc[index]['0'] = df_coef.loc[index]['0']*df_std.loc[index]
                df_divide.loc[index]['0'] = df_coef.loc[index]['0']/df_std.loc[index]
                
    # export coef
    df_multiply.to_csv(os.path.join(path_coef_multiply,str(business_id)+'_time_coefficients_multiply std.csv'), index=True, quoting=1)
    df_divide.to_csv(os.path.join(path_coef_divide,str(business_id)+'_time_coefficients_divide std.csv'), index=True, quoting=1)
    

# In[]
""" --- standardize simulated coef by multiplying or dividing std. of each variable --- """

for business_id in tqdm(list_business):
    for count in range(1,101):
        # import business dataframe
        with open(os.path.join(path_sim_df,'df_'+str(business_id)+'_time_simulation_'+str(count)+'.csv'),'r', encoding = 'utf-8') as file:
            df_business = pd.read_csv(file,header=0)
            file.close()
        df_std = df_business.describe().loc['std']
        # import coef
        with open(os.path.join(path_sim_coef,str(business_id)+'_simulation_'+str(count)+'_time_coefficients.csv'),'r', encoding = 'utf-8') as file:
            df_coef = pd.read_csv(file,header=1,index_col=0)
            file.close()        
        # coef multiply std && coef divide std
        df_multiply = pd.DataFrame()
        df_divide = pd.DataFrame()        
        for index in df_coef.index.values:
            df_multiply = df_multiply.append(df_coef.loc[index])
            df_divide = df_divide.append(df_coef.loc[index])
            # if non-zero coef
            if 'decay' in index and df_coef.loc[index]['0'] > 0:
                if 'stars.' in index:
                    index_std = index[0:len(index.split('.')[0])]+'*'+index[len(index.split('.')[0])+1:len(index)]
                    df_multiply.loc[index]['0'] = df_coef.loc[index]['0']*df_std.loc[index_std]
                    df_divide.loc[index]['0'] = df_coef.loc[index]['0']/df_std.loc[index_std]
                else:
                    df_multiply.loc[index]['0'] = df_coef.loc[index]['0']*df_std.loc[index]
                    df_divide.loc[index]['0'] = df_coef.loc[index]['0']/df_std.loc[index]   
                    
        # export coef
        df_multiply.to_csv(os.path.join(path_sim_coef_multiply,str(business_id)+'_simulation_'+str(count)+'_time_coefficients_multiply std.csv'), index=True, quoting=1)
        df_divide.to_csv(os.path.join(path_sim_coef_divide,str(business_id)+'_simulation_'+str(count)+'_time_coefficients_divide std.csv'), index=True, quoting=1)

