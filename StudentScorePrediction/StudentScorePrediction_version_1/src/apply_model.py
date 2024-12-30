# apply_model.py

# This file is to implement the best model to the train and test dataset

#############################################################################
### Import Modules
#############################################################################

# Essential libraries
import numpy as np
import pandas as pd
import math
import scipy
import random
import datetime
from datetime import datetime



## SciKit Learning Preprocessing  
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


## Pipeline and Train Test Split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

## SciKit Learn ML Models
from sklearn import linear_model, tree, svm, neighbors, ensemble

# Linear Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# Decision Tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Support Vector Machine
from sklearn.svm import SVR, SVC

# Emsemble
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

## Importing xgboost
try:
    import xgboost
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    print("xgboost is not installed.")


## Grid Search
from sklearn.model_selection import GridSearchCV

## Metrics
from sklearn.metrics import accuracy_score

## Additional Tools
import os
import sys
import itertools 
from itertools import product
import joblib
import yaml
from yaml import Loader
import logging

## Custom Helper Function
from myHelper import create_log_file, log_and_print, yaml_pipe_convert, check_make_folder

# Check folder
check_make_folder()

###############################################################
#### Start Logging
###############################################################
# Set logging prefix
header_prefix = 'apply_model' 
create_log_file(header_prefix)
log_and_print('Begin Apply Model Run')   
log_and_print('----------------------------------------------------------------') 


#############################################################################
#### Import Config File
#############################################################################
try:
    with open("apply_model_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
except Exception as e:
        log_and_print(f"Failed to load configuration file. Error: {e}")
        log_and_print(f"Exit Program")
        sys.exit()

#############################################################################
#### Load Data and Parameters
#############################################################################

# Load the dataset
try:
    X_train = pd.read_csv('./data/X_train.csv', index_col='student_id')
    y_train = pd.read_csv('./data/y_train.csv', index_col='student_id')
    X_test = pd.read_csv('./data/X_test.csv', index_col='student_id')
    y_test = pd.read_csv('./data/y_test.csv', index_col='student_id')
except Exception as e:
    log_and_print(f"Failed to load data file. Error: {e}")
    log_and_print(f"Exit Program")
    sys.exit()


# Load Configuration
best_model_pkl_list = config['BEST_MODELS_PKL']
best_model_list = config['BEST_MODELS_LIST']

# Process Best PKL List
pkl_path_list = []
folder_path = './pkl/' 
for each_model in best_model_pkl_list:
    path_str = folder_path + 'best_' + each_model + '.pkl'
    #print(path_str)
    pkl_path_list.append(path_str)


#############################################################################
#### Apply Model to Test Data
#############################################################################
log_and_print("Applying Models from PKL File:")
log_and_print('----------------------------------------------------------------') 

if not best_model_pkl_list:
    log_and_print("No pkl file selected!")
else:
    for model_file in pkl_path_list:
        try:
            best_model = joblib.load(model_file)
        except Exception as e:
            log_and_print(f'Error loading PKL file')
            log_and_print(f'{e}')
            log_and_print('-----------------------------------------------------------------------')
            continue
        best_model.fit(X_train, y_train.values.ravel())
        #log_and_print('Models:', best_model[-1])
        log_and_print(f'Pipe: {best_model}')
        #log_and_print('Params:', best_model.get_params())
        log_and_print(f'R² Score for Training Dataset: {best_model.score(X_train, y_train)}')
        log_and_print(f'R² Score for Test Dataset: {best_model.score(X_test, y_test)}')
        log_and_print('-----------------------------------------------------------------------')

log_and_print(' ')
log_and_print("Applying Models from Manual Selection:")
log_and_print('----------------------------------------------------------------')


models = config['MODELS']
params_list = config['PARAMS']
pipe_list = yaml_pipe_convert(best_model_list, models) 

if not best_model_list:
    log_and_print("No manual model selected!")
else:
    for model_name in best_model_list:
        #print("Applying model: ", model_name)
        pipeline = Pipeline(pipe_list.get(model_name))
        params = params_list.get(model_name)
        
        try:
            pipeline.set_params(**params)
        except Exception as e:
            log_and_print(f"Failed to set parameters.")
            log_and_print(f"Error: {e}")
            log_and_print('----------------------------------------------------------------')
            continue

        try:
            pipeline.fit(X_train, y_train.values.ravel())
        except Exception as e:
            log_and_print(f"Failed to fit the model.")
            log_and_print(f"Error: {e}")
            log_and_print('----------------------------------------------------------------')
            continue    
            
        log_and_print(f"Pipe: {pipeline}")
        log_and_print(f'R² Score for Training Dataset: {pipeline.score(X_train, y_train)}')
        log_and_print(f'R² Score for Test Dataset: {pipeline.score(X_test, y_test)}')
        log_and_print('-----------------------------------------------------------------------')

log_and_print('End of Apply Model Run') 