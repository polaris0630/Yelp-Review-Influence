##################################
##################################
###      Yelp_elastic net      ###
##################################
##################################


# import library
library(glmnet)
library(stringr)
library(tidyverse)

# setup working directory
setwd("C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/yelp_filtered set/simulations_dataframes")

list_csv = dir("C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/yelp_filtered set/simulations_dataframes")

list_result = dir("C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/yelp_filtered set/simulations_coefficients")

# lasso model with penalty.factor
set.seed(2019)

for (i in 1:length(list_csv)) {
  # check if '.csv' or 'prediction':
  if (str_detect(list_csv[i],'.csv') == T){
    if (str_detect(list_csv[i],'predictions') == F){
      
      filename = list_csv[i]
      parameter = str_remove(str_remove(filename,'df_'),'.csv')
      
      # get knot_base
      if (str_detect(parameter,'_order')==T){
        knot_base = 'order'
        print(paste('knot_base is',knot_base))
      }
      else {
        knot_base = 'time'
        print(paste('knot_base is',knot_base))
      }
      # get run number
      run_num = str_extract_all(str_split(string = list_csv[i],pattern = '_simulation')[[1]][2],"\\d+")[[1]]
      
      # get business_id
      business_id = str_remove(parameter,str_c('_',knot_base))
      print(paste('business_id is', business_id))
      print(paste('run number is', run_num))
      
      # haven't been processed:
      if ((paste(business_id,'_',knot_base,'_coefficients.csv',sep = '') %in% list_result) == F){
        
        # import csv file
        data_business = read.csv(filename)
        list_variables = colnames(data_business)[which(str_detect(colnames(data_business),'decay')==TRUE)]
        list_bspline = colnames(data_business)[which(str_detect(colnames(data_business),'spline')==TRUE)]
        
        data_business_x = as.matrix(data_business[,c(list_variables,list_bspline)])
        data_business_y = as.matrix(data_business['stars'])
        
        # setup penalty factor
        p.fac = c(rep(1,length(list_variables)),rep(0,length(list_bspline)))
        
        lasso = cv.glmnet(data_business_x, data_business_y, penalty.factor = p.fac,alpha = 0.8, nfolds=5,intercept=FALSE)
        
        # get coefficients
        coef = as.matrix(coef(lasso,s="lambda.min"))
        #print(coef)
        
        # get prediction
        data_business_pred = predict(lasso, data_business_x)
        
        # save prediction into data_business
        data_business["prediction"] = data_business_pred 
        
        # export data_business coefficients into csv
        # may not save prediction results for simulation due to the limited space
        #write.csv(data_business, file = paste('C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/simulation_results/v2/','df_',business_id,'_',knot_base,'_with predictions.csv',sep = ''),row.names=FALSE)
        write.csv(coef,file = paste('C:/Users/Yichen Jiang/Documents/PHD LIFE/Research/Hawkes Processes/Yelp/yelp_filtered set/simulations_coefficients/',business_id,'_',knot_base,'_coefficients.csv',sep = ''))
        
        rm('data_business')
        rm('data_business_x')
        rm('data_business_y')
      }
    }
  }
}


