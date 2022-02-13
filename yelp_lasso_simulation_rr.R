#####################################
#######   RIDGE REGRESSION    #######
#####################################


# import library
library(glmnet)
library(stringr)
library(tidyverse)
library(Metrics)
library(nnet)

# setup working directory
setwd("C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/dataframe_ready_to_use")

list_csv = dir("C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/dataframe_ready_to_use")

list_result = dir("C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/results")


# lasso model with penalty.factor
set.seed(2019)

filename = list_csv[10]

parameter = str_remove(str_remove(filename,'df_'),'.csv')

# get knot_base
if (str_detect(parameter,'_order')==T){
  knot_base = 'order'
  print(paste('knot_base is',knot_base))}else{
    knot_base = 'time'
    print(paste('knot_base is',knot_base))}


# get business_id
business_id = str_remove(parameter,str_c('_',knot_base))
print(paste('business_id is', business_id))

# import csv file
data_business = read.csv(filename)
list_variables = colnames(data_business)[which(str_detect(colnames(data_business),'decay')==TRUE)]
list_bspline = colnames(data_business)[which(str_detect(colnames(data_business),'spline')==TRUE)]

data_business_x = as.matrix(data_business[,c(list_variables,list_bspline)])
data_business_y = as.matrix(data_business['stars'])

data_business_spline = as.matrix(data_business[,c(list_bspline)])


# fit b-spline into ridge regression

# setup penalty factor
p.fac = c(rep(1,length(list_variables)),rep(0,length(list_bspline)))

lambdas <- 10^seq(3, -2, by = -.1)
rr <- cv.glmnet(data_business_x, data_business_y, lambda = lambdas, penalty.factor = p.fac, alpha = 0, nfolds=5,intercept=FALSE)


preds <- predict(rr, data_business_x,type = 'response')

table(data_business_y)

total <- cbind(preds,data_business_y)
