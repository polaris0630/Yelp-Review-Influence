# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:09:30 2020

@author: Yichen Jiang
"""

"""
--- this file is for implementing logistic reg to compare restuarants with sig cofes and restuarants without ---

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
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ast
import statsmodels.api as sm

# In[]

path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

path_coef_1 = os.path.join(path, 'dataframe_interaction_coefficients')
path_coef_2 = os.path.join(path, 'yelp_dataset','coefficients')
path_business_1 = os.path.join(path, 'business with all features')
path_business_2 = os.path.join(path,'yelp_dataset','business with over 500 reviews & 100 reviews per year','businesses with over 500 reviews with text features')

filenames_coef_1 = os.listdir(path_coef_1)
filenames_coef_2 = os.listdir(path_coef_2)
filenames_coef = filenames_coef_1+filenames_coef_2
filenames_business_1 = os.listdir(path_business_1)
filenames_business_2 = os.listdir(path_business_2)
filenames_business = filenames_business_1+filenames_business_2

# In[]
"""
--- import business info ---

"""
dict_info = {}

for filename in tqdm(filenames_business):
    if filename in filenames_business_1:
        path_business = path_business_1
        with open(os.path.join(path_business,filename),'r',encoding='utf-8') as file:
            for line in file.readlines():
                dict_temp = json.loads(line)
                business_id = filename.split('_with')[0]
                dict_info[business_id] = {}
        for info in dict_temp['RestaurantInfo'].keys():
            if info == 'attributes':
                for attribute in ast.literal_eval(dict_temp['RestaurantInfo']['attributes']).keys():
                    if type(ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute]) == str and \
                    ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute][0:2] == "{'" :# should be a dictionary
                        for sub_attribute in ast.literal_eval(ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute]).keys():
                            if type(ast.literal_eval(ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute])[sub_attribute]) == str:
                                dict_info[business_id]['attributes_'+str(attribute)+'_'+str(sub_attribute)] = \
                                ast.literal_eval(ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute])[sub_attribute].replace("'","")
                            else:
                                dict_info[business_id]['attributes_'+str(attribute)+'_'+str(sub_attribute)] = \
                                ast.literal_eval(ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute])[sub_attribute]                                    
                    else:
                        if type(ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute]) == str:
                            dict_info[business_id]['attributes_'+str(attribute)] = ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute].replace("'","")
                        else:
                            dict_info[business_id]['attributes_'+str(attribute)] = ast.literal_eval(dict_temp['RestaurantInfo']['attributes'])[attribute]
            elif info == 'hours':
                if dict_temp['RestaurantInfo']['hours'] == '':
                    continue
                else:
                    for hour in ast.literal_eval(dict_temp['RestaurantInfo']['hours']).keys():
                        dict_info[business_id]['hours_'+str(hour)] = ast.literal_eval(dict_temp['RestaurantInfo']['hours'])[hour]
            else:
                if type(dict_temp['RestaurantInfo'][info]) == str:
                    dict_info[business_id][info] = dict_temp['RestaurantInfo'][info].replace("'","")
                else:
                    dict_info[business_id][info] = dict_temp['RestaurantInfo'][info]
        # update number of reviews for all businesses
        dict_info[business_id]['review_count'] = len(dict_temp['Reviews'])
            
    else: 
        path_business = path_business_2
        with open(os.path.join(path_business,filename),'r',encoding='utf-8') as file:
            for line in file.readlines():
                dict_temp = json.loads(line)
                business_id = filename.split('_with')[0]
                dict_info[business_id] = {}
        for info in dict_temp['RestaurantInfo'].keys():
            if info == 'attributes':
                if type(dict_temp['RestaurantInfo']['attributes']) == dict:
                    for attribute in dict_temp['RestaurantInfo']['attributes'].keys():
                        if type(dict_temp['RestaurantInfo']['attributes'][attribute]) == str and \
                        dict_temp['RestaurantInfo']['attributes'][attribute][0:2] == "{'":
                            for sub_attribute in ast.literal_eval(dict_temp['RestaurantInfo']['attributes'][attribute]).keys():
                                if type(ast.literal_eval(dict_temp['RestaurantInfo']['attributes'][attribute])[sub_attribute]) == str:
                                    dict_info[business_id]['attributes_'+str(attribute)+'_'+str(sub_attribute)] = \
                                    ast.literal_eval(dict_temp['RestaurantInfo']['attributes'][attribute])[sub_attribute].replace("'","")
                                else:
                                    dict_info[business_id]['attributes_'+str(attribute)+'_'+str(sub_attribute)] = \
                                    ast.literal_eval(dict_temp['RestaurantInfo']['attributes'][attribute])[sub_attribute]
                        else:
                            dict_info[business_id]['attributes_'+str(attribute)] = dict_temp['RestaurantInfo']['attributes'][attribute]
            elif info == 'hours':
                if type(dict_temp['RestaurantInfo']['hours']) == dict:
                    for hour in dict_temp['RestaurantInfo']['hours'].keys():
                        dict_info[business_id]['hours_'+str(hour)] = dict_temp['RestaurantInfo']['hours'][hour]
            else:
                dict_info[business_id][info] = dict_temp['RestaurantInfo'][info]
        # update number of reviews for all businesses
        dict_info[business_id]['review_count'] = len(dict_temp['Reviews'])
        
# In[]
# remove unrecognized character
for business_id in dict_info.keys():
    for info in dict_info[business_id].keys():
        if type(dict_info[business_id][info]) == str and dict_info[business_id][info][0:1] == "u":
            dict_info[business_id][info] = dict_info[business_id][info][1:len(dict_info[business_id][info])]

# dict -> df
df_info = pd.DataFrame.from_dict(dict_info).T  

# replace nan with 'none'
df_info = df_info.where(df_info.notnull(),'none')

# In[]
"""
--- encoding for all attributes ---

"""
          

dict_encode = {}
# list of all attributes that are necessary to be encoded
list_encode = ['attributes_WiFi', 'attributes_RestaurantsAttire', 'attributes_RestaurantsTakeOut',\
               'attributes_GoodForKids', 'attributes_Corkage', 'attributes_RestaurantsGoodForGroups', \
               'attributes_RestaurantsDelivery', 'attributes_HasTV', 'attributes_NoiseLevel', \
               'attributes_Alcohol', 'attributes_Caters', 'attributes_OutdoorSeating', 'attributes_BikeParking',\
               'attributes_BusinessParking_garage','attributes_BusinessParking_street', \
               'attributes_BusinessParking_validated', 'attributes_BusinessParking_lot', \
               'attributes_BusinessParking_valet', 'attributes_BYOBCorkage', \
               'attributes_BusinessAcceptsCreditCards', 'attributes_RestaurantsReservations',\
               'attributes_GoodForMeal_dessert', 'attributes_GoodForMeal_latenight', \
               'attributes_GoodForMeal_lunch', 'attributes_GoodForMeal_dinner', 'attributes_GoodForMeal_brunch',\
               'attributes_GoodForMeal_breakfast', 'attributes_Ambience_romantic','attributes_Ambience_intimate',\
               'attributes_Ambience_classy', 'attributes_Ambience_hipster', 'attributes_Ambience_divey',\
               'attributes_Ambience_touristy', 'attributes_Ambience_trendy', 'attributes_Ambience_upscale',\
               'attributes_Ambience_casual', 'state']

# encoding
for var in list_encode:
    # clear (define) count
    count = 1
    dict_encode[var] = {}
    for index in df_info[var].value_counts().index.values:
        if type(index) == str:
            index = index.strip("'")
        if index != 'none':
            dict_encode[var][index] = count
            count += 1
    if 'none' in df_info[var].value_counts().index.values:
        dict_encode[var]['none'] = count
            
        

""" --- notice that 'attributes_RestaurantsPriceRange2' has already been encoded --- """
""" --- AND 'is_open', 'review_count', 'stars' are numerical var that has to be transformed into numerical form --- """

# list of numerical vars that can be directly used
list_num = ['attributes_RestaurantsPriceRange2', 'is_open', 'stars', 'review_count']

# In[]
"""
--- import coef results and label the data according to whether one business has sig coef or not ---
--- extract other variables

"""
list_var = []

for i in range(0,5):
    list_var.append(str(i+1)+'star')
list_var += ['votes', 'average_stars', 'elite_count', 'review_count', 'fan_count', 'yelping_since', 'mean_prob', 'sentiment_polarity', 'sentiment_subjectivity']

list_var += ['stars*mean_prob','stars*sentiment_polarity','stars*sentiment_subjectivity']

list_temp = ['decay_'+str(var) for var in list_var]
list_var = list_temp
del list_temp

dict_var = {}

for filename in tqdm(filenames_coef):
    if 'time' in filename and 'order' not in filename:
        if filename in filenames_coef_1:
            path_coef = path_coef_1
        else: 
            path_coef = path_coef_2
        with open(os.path.join(path_coef,filename),'r', encoding = 'utf-8') as file:
            df_temp = pd.read_csv(file,header=1,index_col=0)
        business_id = filename.split('_time')[0]
        dict_var[business_id] = {}
        
        # sum up coefs, if zero then label 0, alse label 1
        if df_temp[df_temp.index.str[0:5]=='decay']['0'].sum() == 0.0:
            dict_var[business_id]['label'] = 0
        else:
            dict_var[business_id]['label'] = 1
            
        # number of years
        dict_var[business_id]['year_count'] = len(df_temp[df_temp.index.str[0:8]=='b_spline'])
        
        # labeling all the decay vars
        for var in df_temp[df_temp.index.str[0:5]=='decay']['0'].index:
            if 'label_'+var.strip('_'+var.split('_')[len(var.split('_'))-1]) not in dict_var[business_id].keys():
                dict_var[business_id]['label_'+var.strip('_'+var.split('_')[len(var.split('_'))-1])] = 0
            if df_temp[df_temp.index.str[0:5]=='decay']['0'][var] != 0.0:
                dict_var[business_id]['label_'+var.strip('_'+var.split('_')[len(var.split('_'))-1])] = 1

        # number of significant variables
        df_decay = df_temp[df_temp.index.str[0:5]=='decay']['0'].value_counts()
        dict_var[business_id]['sig_coef_count'] = df_decay[df_decay.index.values != 0.0].sum()
        
        # encoding
        for var in list_encode:
            if type(df_info.loc[business_id][var]) == str:
                dict_var[business_id][var] = dict_encode[var][df_info.loc[business_id][var].strip("'")]
            else:
                dict_var[business_id][var] = dict_encode[var][df_info.loc[business_id][var]]
                
        for var in list_num:
            if var == 'stars':
                dict_var[business_id][var] = float(df_info.loc[business_id][var])
            else:
                if var == 'attributes_RestaurantsPriceRange2':
                    if df_info.loc[business_id][var] == 'none':
                         dict_var[business_id][var] = 0
                    else:
                        dict_var[business_id][var] = int(df_info.loc[business_id][var])
                else:
                    dict_var[business_id][var] = int(df_info.loc[business_id][var])
        
df_var = pd.DataFrame.from_dict(dict_var).T

# In[]
"""
--- logistic regression ---
--- sklearn ---

"""
list_select  = ['stars','review_count','year_count','state','attributes_RestaurantsAttire','attributes_RestaurantsPriceRange2']
# plus

for var in list_encode:
    if 'attributes_GoodForMeal' in var:
        list_select.append(var)

X = df_var[list_select]
y = df_var['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size=0.33, random_state=2020)

LR = LogisticRegression(solver='lbfgs',max_iter=10000)

LR.fit(X_train, y_train)

LR.predict(X_test)

df_lr = pd.DataFrame({'Observations':y_test,'Predictions':LR.predict(X_test)})

metrics.confusion_matrix(df_lr['Observations'],df_lr['Predictions'])

# In[]
"""
--- logistic regression ---
--- statsmodel ---
--- glm binomial ---

"""
glm_binom = sm.GLM(y,X, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())
print(res.predict(X_test, linear = True)) 

# In[]
"""
--- logistic regression ---
--- statsmodel ---
--- logit reg ---

"""

logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())
print(result.predict(X_test, linear = True)) 

#Inverse-logit


# results are not categorical ?????

# In[]
"""
--- check relationship between is_open and year_count ---

"""
df_corr = df_var.corr('pearson')
corr = df_corr['year_count']['is_open'] # low corr coef
print(corr)

# In[]
# histogram
df_count = pd.DataFrame(columns=['year_count','percentage','restaurant_num'])
for i in range(1,20):
    df_temp = df_var[df_var.year_count == float(i)]
    count = len(df_temp)
    if count == 0.0:
        continue
    percentage = len(df_temp[df_temp.is_open == 1.0])/count
    df_count.loc[i] = [float(i),percentage,count]

df_count = df_count.sort_values(by = 'year_count',axis = 0,ascending = False)

# In[]
# label vs. is_open
df_var.label.value_counts()


