# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:22:31 2019

@author: Yichen Jiang
"""

"""
--- this file is for testing if hawkes features are applicable for different models ---

"""
# In[]
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from scipy.interpolate import UnivariateSpline

# data and metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

# regression method
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import LassoCV

# classification method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier

# In[]
createVar = locals()

# In[]
"""
--- read dictonary that saves all the model results ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'
file = open(os.path.join(path,'comparison_model results', 'dict_results.json'),'r',encoding='utf-8')
dict_results = {}
for line in file.readlines():
    dict_results = json.loads(line)

# In[]
"""
--- OR !!! create dictionary that saves all the model results ---

"""
dict_results = {'1 delta':{}, '5 deltas':{}}

# In[]
"""
--- choose business ---

"""
"""
--- choose business ---

"""
business_id = 'DkYS3arLOhA8si5uUEmHOw' # of reviews: 5206
#business_id = '2weQS-RnoOBhb1KsHKyoSQ' # of reviews: 4534
#business_id =  'beuVp5CZxCdNvQIIPBS2rw' # of reviews: 1029
#business_id = 'RESDUcs7fIiihp38-d6_6g' # of reviews: 8568
#business_id = 'ii8sAGBexBOJoYRFafF9XQ' # of reviews: 2557

# In[]
"""
--- import business dataframe ---

"""
path = r'C:\Users\Yichen Jiang\Documents\PHD LIFE\Research\Hawkes Processes\Yelp'

# regular 1 delta
df_business = pd.read_csv(os.path.join(path,'dataframe_ready_to_use' , 'df_'+str(business_id)+'.csv'), index_col=0)

# In[]
# 5 deltas
df_business = pd.read_csv(os.path.join(path,'dataframe_ready_to_use' , 'df_'+str(business_id)+'_5 deltas.csv'), index_col=0)

# In[]












# In[]
"""
--- split data into dependent variable and independent variables

"""
# In[] #x variabls
list_vars_all = []
# In[] # text variables
list_vars_text = ['decay_mean_prob', 'decay_mean_weight', 'decay_sentiment_polarity', 'decay_sentiment_subjectivity', 'decay_text_length', 'decay_textclean_length']
list_vars_all += list_vars_text
# In[] # user variables
list_vars_user = ['decay_friend_count', 'decay_elite_count', 'decay_review_count', 'decay_fan_count', 'decay_yelping_since']
list_vars_all += list_vars_user
# In[] # star decays
list_vars_star = ['decay_1star','decay_2star','decay_3star','decay_4star','decay_5star']
list_vars_all += list_vars_star
# In[] # other variables
list_vars_other = ['decay_average_stars','decay_cool','decay_funny','decay_useful']
list_vars_all += list_vars_other
# In[]
df_business_x = df_business[list_vars_all]

df_business_y = df_business['stars'].astype(np.float64)

# In[]
""" for multi deltas """
# all variables with all deltas
list_delta = [0.005, 0.1, 1, 5, 10]
list_vars = []
for variable in list_vars_all:
    for delta in list_delta:
        list_vars.append(str(variable)+'_'+str(delta))
        
# In[]
df_business_x = df_business[list_vars]

df_business_y = df_business['stars'].astype(np.float64)

# In[]











# In[]
"""
--- split data into test set and training set ---

"""
X = df_business_x
y = df_business_y.astype(np.int)

# set proportion of test set
proportion = 0.2

X_train,X_test, y_train, y_test = train_test_split(df_business_x,df_business_y,test_size = proportion,random_state=100)
# check data proportion and structure
print ('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,y_train.shape, X_test.shape,y_test.shape))

# In[]











# In[]
"""
--- implement data into different regression models ---

"""
""" define performance measurements """
scorers = {
        'mean_squared_error': make_scorer(mean_squared_error),
        'r2_score': make_scorer(r2_score)
    }

""" define the model and the parameters you want to tune """
# linear regression
linreg = LinearRegression()
linreg_parameters = {
        }
# lasso regression
lassoreg = LassoCV(alphas = [1, 0.1, 0.05, 0.001, 0.0005, 0.00045, 0.0004, 0.0003, 0.0002, 0.0001, 0.00005],random_state=100,cv=10)
lassoreg_parameters = {'alphas':[0.1,0.05,0.001,0.0005,0.00045,0.0004,0.0003,0.0002,0.0001,0.00005,0],
                       'normalize':(True,False)
        }

# In[]
# choose model with its parameters
model_name = lassoreg
parameters = lassoreg_parameters
model_name_str = str(model_name).split('(')[0]
# choose the measurement
refit_score = 'r2_score'
# choose number of fold
num_fold = 10
skf = StratifiedKFold(n_splits=num_fold, random_state = 100)
# build the model
#model = GridSearchCV(model_name,parameters,cv=skf)
model = model_name
# build the model
reg = model.fit(X_train, y_train)

# detect num of delta
delta = ''
if len(X.columns) < 80:
    delta = '1 delta'
else: delta = '5 deltas'

if business_id not in dict_results[delta].keys():
    dict_results[delta][business_id] = {}
if model_name_str not in dict_results[delta][business_id] .keys():
    dict_results[delta][business_id][model_name_str] = {}
if 'test_size='+str(proportion) not in dict_results[delta][business_id][model_name_str].keys():
    dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)] = {}
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['test_size'] = proportion
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['business_id'] = business_id
#dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['best_parameters'] = reg.best_params_
#dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['refit_score'] = refit_score # evaluationn measurement
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['k_fold'] = num_fold # evaluationn measurement
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['model'] = str(model)

# In[]
"""
--- alpha ---

"""
reg.alpha_
# save alpha
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['alpha'] = reg.alpha_

# In[]
"""
--- coefficients ---

"""
reg.coef_

reg_coef = pd.Series(reg.coef_, index = X_train.columns)
print("Linear regression picked " + str(sum(reg_coef != 0)) + " variables and eliminated the other " +  str(sum(reg_coef == 0)) + " variables")
# save coefficients
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['coefficients'] = [[reg_coef.index[i],reg_coef[i]] for i in range(len(reg_coef))]

# In[]
"""
--- visualize coefficients ---

"""
imp_coef = pd.concat([reg_coef.sort_values()])
plt.rcParams['figure.figsize'] = (15, 15)
plt.rcParams['font.size'] = 25
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Linear Regression Model")   

# In[]:
"""
--- residuals  training set---

"""
plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams['font.size'] = 25
reg_preds_train = pd.DataFrame({'preds':reg.predict(X_train), 'true':y_train}) 
reg_preds_train['residuals'] = reg_preds_train['true'] - reg_preds_train['preds']
reg_preds_train.plot(x = 'preds', y = 'residuals',kind = 'scatter')

# In[]
"""
--- R^2   training set---

"""
r2_train = r2_score(reg_preds_train['true'], reg_preds_train['preds'])
print(r2_train)

# save r2 score
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['r2_score_trian'] = r2_train

# In[]
"""
--- mean square error   training set ---

"""
mse_train = mean_squared_error(reg_preds_train['true'], reg_preds_train['preds'])
print(mse_train)

# save mse
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['mse_trian'] = mse_train

# In[]
"""
--- root mean square error training set ---

"""
rmse_train = math.sqrt(mse_train)
print(rmse_train)

# save rmse
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['rmse_trian'] = rmse_train

# In[]
"""
--- residuals   test set ---

"""
plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams['font.size'] = 25
reg_preds_test = pd.DataFrame({'preds':reg.predict(X_test), 'true': y_test})
reg_preds_test['residuals'] = reg_preds_test['true'] - reg_preds_test['preds']
reg_preds_test.plot(x = 'preds', y = 'residuals', kind = 'scatter')

# In[]
"""
--- R^2   test set ---

"""
r2_test = r2_score(reg_preds_test['true'], reg_preds_test['preds'])
print(r2_test)

# save r2 score
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['r2_score_test'] = r2_test

# In[]
"""
--- mean square error   test set ---

"""
mse_test = mean_squared_error(reg_preds_test['true'], reg_preds_test['preds'])
print(mse_test)

# save mse
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['mse_test'] = mse_test

# In[]
"""
--- root mean square error test set ---

"""
rmse_test = math.sqrt(mse_test)
print(rmse_test)

# save rmse
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['rmse_test'] = rmse_test

# In[]
"""
--- save dict_results ---

"""
path_results = os.path.join(path,'comparison_model results')
with open(os.path.join(path,'comparison_model results', 'dict_results.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_results, outfile, ensure_ascii=False) 
    
# In[]














# In[]
"""
--- upsample & downsample ---

"""
# get number of samples and create dataframes for each dataset
for star in range(0,5):
    createVar['num_'+str(star+1)+'star'] = len(df_business[df_business['stars'] == star+1])
    createVar['df_business_'+str(star+1)+'star'] = df_business[df_business['stars'] == star+1]

# In[]
# set up number of samples
num_sample = 2000

for star in range(0,5):
    # num of samples in the dataset < target num: taking samples with replacement
    if createVar['num_'+str(star+1)+'star'] < num_sample:
        replacement = True
    else: replacement = False
    X_temp, y_temp = \
    resample(createVar['df_business_'+str(star+1)+'star'][list_vars_all], createVar['df_business_'+str(star+1)+'star']['stars'].astype(np.int), n_samples=num_sample, replace=replacement,random_state=50)
    # combine dataset inn different label
    if star+1 == 1:
        X = X_temp
        y = y_temp
    else:
        X = pd.concat([X, X_temp])
        y = pd.concat([y, y_temp])
        
# In[]
"""
--- split data into test set and training set ---

"""
# set proportion of test set
proportion = 0.2
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = proportion,random_state=100)
# check data proportion and structure
print ('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,y_train.shape, X_test.shape,y_test.shape))

# In[]
"""
--- implement data into different classification models ---

"""
""" define performance measurements """
scorers = {
    #'precision_score': make_scorer(precision_score,average='weighted'),
    #'recall_score': make_scorer(recall_score,average='weighted'),
    'accuracy_score': make_scorer(accuracy_score)
    }

""" define the model and the parameters you want to tune """
# random forest
rf = RandomForestClassifier(random_state=0,n_jobs=-1)
rf_parameters = {
        'n_estimators':[10,100,1000], 
        'max_depth':[None,1,2,3], 
        'max_features':('sqrt','log2', None)
        }
# support vector machine
svm = SVC(random_state=0)
svm_parameters = {'gamma':('auto','scale'),
                  'kernel':('linear','poly','rbf','sigmoid','precomputed'),
                  'decision_function_shape':('ovo','ovr',None)
        }
# k nearest neighbors
knn = KNeighborsClassifier(n_jobs=-1)
knn_parameters = {'n_neighbors':[3,4,5,6,7,8],
                  'weights':('uniform','distance'),
                  'leaf_size':[30,40,50],
                  'p':[1,2]
        }
# logistic regression
lg = LogisticRegressionCV(random_state=0,class_weight='balanced')
lg_parameters = {'penalty':('l1','l2','elasticnet'),
                 'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
                 'multiclass':('ovr','multinomial')
        }
# linear classifiers (SVM, logistic regression...) with SGD training
sgd = SGDClassifier(learning_rate='optimal',n_jobs=-1)
sgd_parameters = {'loss':('hinge','log','modified_huber'),
                  'penalty':('l1','l2','elasticnet'),
                  'max_iter':[100,200,500,1000,2000],
                  'alpha':[0.00001,0.00002,0.00005,0.0001]
        }

# In[]
"""
--- choose the model and set up the parameters with possible tuning options ---

"""
# choose model with its parameters
model_name = svm
parameters = svm_parameters
model_name_str=str(model_name_str).split('(')[0]
# choose the measurement
refit_score = 'accuracy_score' 
# choose number of fold
num_fold = 10
skf = StratifiedKFold(n_splits=num_fold, random_state = 100)

# build the model
model = GridSearchCV(model_name,parameters,scoring=scorers,refit=refit_score,cv=skf)

# train the model with training data
clf = model.fit(X_train, y_train)

# saving model information
delta = ''
if len(X.columns) < 80:
    delta = '1 delta'
else: delta = '5 deltas'

if business_id not in dict_results[delta].keys():
    dict_results[delta][business_id] = {}
if model_name_str not in dict_results[delta][business_id] .keys():
    dict_results[delta][business_id][model_name_str] = {}
if 'test_size='+str(proportion) not in dict_results[delta][business_id][model_name_str].keys():
    dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)] = {}
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['test_size'] = proportion
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['business_id'] = business_id
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['best_parameters'] = clf.best_params_
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['refit_score'] = refit_score # evaluationn measurement
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['k_fold'] = num_fold # numer of folds
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['model'] = str(model)

# In[]:
"""
--- print the best combination of parameters ---

"""
clf.best_params_

# In[]
"""
--- confusion matrix of training set ---

"""
clf_preds_train = pd.DataFrame({'preds':clf.predict(X_train), 'true':y_train})

print(confusion_matrix(clf_preds_train['true'],clf_preds_train['preds']))

# saving confusion matrix
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['confusion_matrix_train'] = \
str(confusion_matrix(clf_preds_train['true'],clf_preds_train['preds']))

# In[]
"""
--- accuracy of training set ---

"""
print(accuracy_score(clf_preds_train['true'],clf_preds_train['preds']))
# saving accuracy score
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['accuracy_score_train'] = accuracy_score(clf_preds_train['true'],clf_preds_train['preds'])

# In[]
"""
---confusion matrix of test set ---

"""
clf_preds_test = pd.DataFrame({'preds':clf.predict(X_test), 'true':y_test})

print(confusion_matrix(clf_preds_test['true'],clf_preds_test['preds']))
# saving confusion matrix
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['confusion_matrix_test'] = \
str(confusion_matrix(clf_preds_test['true'],clf_preds_test['preds']))

# In[]
"""
--- accuracy of test set ---

"""
print(accuracy_score(clf_preds_test['true'],clf_preds_test['preds']))
# saving accuracy score
dict_results[delta][business_id][model_name_str]['test_size='+str(proportion)]['accuracy_score_test'] = accuracy_score(clf_preds_test['true'],clf_preds_test['preds'])

# In[]











# In[]
"""
--- save dict_results ---

"""
path_results = os.path.join(path,'comparison_model results')
with open(os.path.join(path,'comparison_model results', 'dict_results.json'), 'w+', encoding="utf-8") as outfile:
    json.dump(dict_results, outfile, ensure_ascii=False) 

# In[]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# In[]
"""
--- backup code ---

"""

for delta in dict_results.keys():
    for business_id in dict_results[delta].keys():
        for key in dict_results[delta][business_id].keys():
            print(key)
