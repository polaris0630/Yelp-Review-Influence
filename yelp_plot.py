# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:38:45 2020

@author: Yichen Jiang
"""

"""
--- this file is for pre-processing coefs & sd. for each variable &creating cleveland plot on coefs data ---
--- and more other plots ---

"""
# In[]
import numpy as np
import math
import os
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ast
import statsmodels.api as sm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#import warnings; warnings.filterwarnings(action='once')



# Version
print(mpl.__version__)  #> 3.0.0
print(sns.__version__)  #> 0.9.0

# In[]
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'
#path_coef = os.path.join(path,'yelp_filtered set','coefficients')
#path_coef_sim  = os.path.join(path,'yelp_filtered set','simulations_coefficients')

#path_coef = os.path.join(path,'yelp_filtered set','coefficients multiply std')
#path_coef_sim  = os.path.join(path,'yelp_filtered set','simulations_coefficients multiply std')

path_coef = os.path.join(path,'yelp_filtered set','coefficients')
path_coef_sim  = os.path.join(path,'yelp_filtered set','simulations_coefficients')


filenames_coef = os.listdir(path_coef)
filenames_coef_sim = os.listdir(path_coef_sim)

# In[]
# import list of businesses (with over 100 reviews/year)
df_list = pd.read_csv(os.path.join(path,'yelp_filtered set','business with 100 reviews per year in average and non-zero coef_lasso.csv'),index_col = 0)
list_business = list(df_list.index.values)

# dict matching business_id and filename
dict_match = {}
for filename in filenames_coef:
    business_id = filename.split('_time')[0]
    if business_id not in dict_match.keys():
        dict_match[business_id] = filename

# In[]
"""
--- compute sd. for each business on each variable ---

"""

# In[]
"""
--- import coefs from original coef set (1715 businesses in total) ---

"""
dict_business_original = {}
dict_business_sim = {}

for business_id in tqdm(dict_match.keys()):
    dict_business_original[business_id] = {}
    if 'multiply' in path_coef_sim or 'divide' in path_coef_sim:
        header_num = 0
    else: header_num = 1
    
    # lasso(original)
    with open(os.path.join(path_coef,dict_match[business_id]),'r', encoding = 'utf-8') as file:
        df_coef = pd.read_csv(file,header=header_num,index_col=0)
        file.close()
        for index in df_coef.index:
            if 'decay' in index: # get coefficients
                dict_business_original[business_id][index] = df_coef.loc[index]['0']

        
    # lasso simulation
    if business_id in list_business: # if current business has been simulated
        # remove coefficients of b-spline
        index_bspline = list(df_coef.index.values).index('b_spline_0')
        df_coef = df_coef.iloc[:index_bspline,:]
        for count in range(1,101):
            if 'multiply' in path_coef_sim:
                filename = str(business_id)+'_simulation_'+str(count)+str('_time_coefficients_multiply std.csv')
            elif 'divide' in path_coef_sim:
                filename = str(business_id)+'_simulation_'+str(count)+str('_time_coefficients_divide std.csv')
            else:
                filename = str(business_id)+'_simulation_'+str(count)+str('_time_coefficients.csv')
            
            df_temp = pd.read_csv(os.path.join(path_coef_sim,filename),header=header_num,index_col=0)
            # remove coefficients of b-spline
            df_temp = df_temp.iloc[:index_bspline,:]
            df_coef = pd.concat([df_coef,df_temp],axis=1)
        
        # revise column names
        df_coef.columns = ['real']+list(range(0,count))
    
        """ --- calculate test statistics --- """
        """ --- record coef values --- """
        dict_business_sim[business_id] = {}
        dict_business_sim[business_id]['run_num'] = count
        
        for variable in df_coef.index:
            list_coef = []
            dict_business_sim[business_id][variable] = {}
            real_coef = abs(df_coef.loc[variable,'real'])
            count = 0
            for column in df_coef.columns:
                if column == 'real':
                    continue
                else:
                    list_coef.append(df_coef.loc[variable,column])
                    coef = abs(df_coef.loc[variable,column])
                    if coef >= abs(real_coef):#real_coef:
                        count += 1
            dict_business_sim[business_id][variable]['count'] = count
            dict_business_sim[business_id][variable]['p-value'] = \
            dict_business_sim[business_id][variable]['count']/dict_business_sim[business_id]['run_num']
            dict_business_sim[business_id][variable]['coefs'] = list_coef
            
# In[]
"""
--- classify original & simulated coef data set ---
--- non-zero coefs ---
--- dict -> df ---

"""
dict_coef = {}
list_decay = ['0.005','0.05','0.1','1','5']
list_variable = ['decay_1star','decay_2star','decay_3star','decay_4star','decay_5star',
                 'decay_average_stars','decay_elite_count','decay_fan_count','decay_mean_prob',
                 'decay_review_count','decay_sentiment_polarity','decay_sentiment_subjectivity',
                 'decay_stars.mean_prob','decay_stars.sentiment_polarity',
                 'decay_stars.sentiment_subjectivity','decay_votes','decay_yelping_since']
count = 0
for business_id in tqdm(dict_business_original.keys()):
    for variable in dict_business_original[business_id].keys():
        if dict_business_original[business_id][variable] == 0.0:
            continue
        else:
            if business_id in list_business:
                data_type_2 = 'selected'
            else:
                data_type_2 = 'non-selected'
            dict_coef[count] = {'business_id': business_id, 'coef': float(dict_business_original[business_id][variable]),
                              'variable with decay': variable, 
                              'variable': variable.strip('_'+variable.split('_')[len(variable.split('_'))-1]),
                              'decay':variable.split('_')[len(variable.split('_'))-1],
                              'data_type_1': 'original', 'data_type_2': data_type_2,
                              'p-value overall': None, 'p-value specific': None,
                              'significance overall': None, 'significance specific': None}
            count += 1

for business_id in tqdm(dict_business_sim.keys()):
    for variable in list_variable:
        # find the lowest p-value for a variable among all decays
        pvalue_o = 1.0
        for decay in list_decay:
            if dict_business_sim[business_id][str(variable)+'_'+str(decay)]['p-value'] < pvalue_o:
                pvalue_o = dict_business_sim[business_id][str(variable)+'_'+str(decay)]['p-value']
        
        for decay in list_decay:
            for coef in dict_business_sim[business_id][str(variable)+'_'+str(decay)]['coefs']:
                if coef == 0.0:
                    continue
                else:
                    # check significance of coef asscording to p-value
                    if dict_business_sim[business_id][str(variable)+'_'+str(decay)]['p-value'] <= 0.05:
                        sig_s = 'significant'
                    else:
                        sig_s = 'non-significant'
                    if pvalue_o <= 0.05:
                        sig_o = 'significant'
                    else:
                        sig_o = 'non-significant'
                        
                    dict_coef[count] = {'business_id': business_id, 'coef': float(coef),
                                      'variable with decay': str(variable)+'_'+str(decay), 'variable': variable,
                                      'decay': decay, 'data_type_1': 'simulation', 'data_type_2': 'selected',
                                      'p-value overall': pvalue_o, 
                                      'p-value specific': dict_business_sim[business_id][str(variable)+'_'+str(decay)]['p-value'],
                                      'significance overall': sig_o, 'significance specific': sig_s}
                    count += 1


df_coef = pd.DataFrame.from_dict(dict_coef).T
df_coef['coef'] = pd.to_numeric(df_coef['coef'])

df_coef = df_coef.sort_values(by=['variable'])

# In[]
""" --- remove 'decay_' for all variable & variable with decay --- """
for index in tqdm(df_coef.index.values):
    df_coef['variable'].iloc[index] = df_coef['variable'].iloc[index][6:len(df_coef['variable'].iloc[index])]
    df_coef['variable with decay'].iloc[index] = df_coef['variable with decay'].iloc[index][6:len(df_coef['variable with decay'].iloc[index])]
    
# In[]
""" --- settings for plots --- """
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# In[]
"""
--- define new index by ordering variable types ---
--- Star-rating features ---
--- user features ---
--- text features ---

--- AND !!! adjust variable name, upper/lowercase, and character ---

"""
# adjust variable order & name
dict_index = {'1star':'1 Star', '2star':'2 Star', '3star':'3 Star', '4star':'4 Star', \
              '5star': '5 Star', 'average_stars': 'Average Star-Rating', 'votes': 'Votes', \
              'elite_count': 'Elite Count', 'fan_count': 'Fan Count', 'review_count': 'Review Count', \
              'yelping_since': 'Yelping_Since', 'mean_prob': 'Average Word Probability', \
              'sentiment_polarity': 'Sentiment: Polarity', \
              'sentiment_subjectivity': 'Sentiment: Subjectivity', \
              'stars.mean_prob': 'Stars * Average Word Probability', \
              'stars.sentiment_polarity': 'Stars * Polarity', \
              'stars.sentiment_subjectivity': 'Stars * Subjectivity'}

# In[]

""" 
--- categorical dot plot for ORIGINAL dataset ---
--- arange the dataframe with specific order ---

"""
df_data = pd.DataFrame()
df_temp = df_coef[df_coef.data_type_1=='original']

for var in tqdm(dict_index.keys()):
    for index in df_temp.index:
        if df_temp.loc[index]['variable'] == var:
            df_data = df_data.append(df_temp.loc[index])
            # update var name
            df_data.loc[index, 'variable'] = dict_index[var]

# In[]
"""
--- categorical dot plot for ORIGINAL dataset ---
--- specify variable without decay ---
--- by each coef/business (each coef corresponds to 1 business) ---

"""
""" --- dot plot --- """

ax = sns.catplot(x="coef", y="variable",
            data=df_data,
            height=12, aspect=1.5,palette="Set1")
plt.xlim(-15,15)
plt.xlabel('Coefficients')
plt.ylabel('Variables')
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
""" --- violin plot --- """

plt.figure(figsize=(20, 15))
sns.violinplot(x="coef", y="variable",
                    data=df_coef[df_coef.data_type_1=='original'],
                    scale="width", palette="Set1", 
                    width=1,cut=0)
plt.xlim(-15,15)
plt.xlabel('Coefficients')
plt.ylabel('Variables')
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
""" --- count plot --- """
plt.figure(figsize=(20, 15))
sns.violinplot(x="coef", y="variable",
                    data=df_coef[df_coef.data_type_1=='original'],
                    scale="count", palette="Set1", 
                    width=1, cut=0)
plt.xlim(-15,15)
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]

""" 
--- categorical dot plot for SELECTED ORIGINAL dataset ---
--- arange the dataframe with specific order ---

"""
df_data = pd.DataFrame()
df_temp = df_coef[(df_coef.data_type_1 == 'original')&(df_coef.data_type_2 == 'selected')]

for var in tqdm(dict_index.keys()):
    for index in df_temp.index:
        if df_temp.loc[index]['variable'] == var:
            df_data = df_data.append(df_temp.loc[index])
            # update var name
            df_data.loc[index, 'variable'] = dict_index[var]



# In[]
"""
--- categorical dot plot for SELECTED ORIGINAL dataset ---
--- specify variable without decay ---
--- by each coef/business (each business corresponds to 1 coef) ---

"""
""" --- dot plot --- """

ax = sns.catplot(x='coef', y='variable', 
            data=df_data,
            height=12, aspect=1.5,palette="Set1")
plt.xlabel('Coefficients')
plt.ylabel('Variables')

sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
""" --- violinplot --- """
plt.figure(figsize=(20, 15))
sns.violinplot(x="coef", y="variable",
               data=df_coef[(df_coef.data_type_1 == 'original')&(df_coef.data_type_2 == 'selected')],
               scale="width", palette="Set1", 
               width=1)
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
"""
--- categorical dot plot for all SIMULATION data ---
--- with overall significance ---
--- by each coef (each business corresponds to 1 or more coefs (100-runs))

"""
""" --- dot plot --- """

ax = sns.catplot(x='coef', y='variable', hue='significance overall',
            data=df_coef[df_coef.data_type_1=='simulation'],
            height=12, aspect=1.5,palette="Set1")
plt.xlim(-15,15)
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
""" --- violin plot --- """

plt.figure(figsize=(20, 15))
sns.violinplot(x="coef", y="variable", hue='significance overall',
                    data=df_coef[df_coef.data_type_1=='simulation'],
                    scale="width", palette="Set1", split=True,
                    width=1,cut=0)
plt.xlim(-15,15)
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
"""
--- categorical dot plot for 156 SELECTED businesses with ORIGINAL & SIMULATION ---
--- with overall significance ---

"""
""" --- re-arrange data --- """

df_data = pd.DataFrame()
df_temp = df_coef[df_coef.data_type_2=='selected']

for var in tqdm(dict_index.keys()):
    for index in df_temp.index:
        if df_temp.loc[index]['variable'] == var:
            df_data = df_data.append(df_temp.loc[index])
            # update var name
            df_data.loc[index, 'variable'] = dict_index[var]


# In[]
"""
--- categorical dot plot for 156 SELECTED businesses with ORIGINAL & SIMULATION ---
--- with overall significance ---

"""
""" --- dot plot ---"""

ax = sns.catplot(x='coef', y='variable', hue='data_type_1',
            data=df_coef[df_coef.data_type_2=='selected'],
            height=12, aspect=1.5,palette="Set1")
plt.xlabel('Coefficients')
plt.ylabel('Variables')
plt.xlim(-15,15)
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
""" --- violin plot ---"""
plt.figure(figsize=(20, 15))
sns.violinplot(x="coef", y="variable", hue='data_type_1',
                    data=df_data,
                    scale="width", palette="Set1", split=True,
                    width=1)
plt.xlabel('Coefficients')
plt.ylabel('Variables')
plt.xlim(-15,15)
plt.legend(loc='lower right')
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
"""
--- categorical dot plot for coefs from all businesses (original & simulation) ---
--- with data_type_1 ---

"""
""" --- dot plot --- """
ax = sns.catplot(x='coef', y='variable', hue='data_type_1',
            data=df_coef,
            height=12, aspect=1.5,palette="Set1")
plt.xlim(-20,20)
sns.set()
sns.set_context('notebook', font_scale=2)

# In[]
""" --- violin plot ---"""
plt.figure(figsize=(20, 15))
sns.violinplot(x="coef", y="variable", hue='data_type_1',
                    data=df_coef,
                    scale="width", palette="Set1", split=True,
                    width=1,cut=0)
plt.xlim(-20,20)
sns.set()
sns.set_context('notebook', font_scale=2)



