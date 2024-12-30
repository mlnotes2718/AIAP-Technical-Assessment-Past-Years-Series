# evaluate_regression_models.py

### Disclaimer ##############################################################
### The original code is generated from ChatGPT                        ######
### However, the code is heavily modified for grid search evaluation   ######
#############################################################################

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
header_prefix = 'gs_model_eval' 
create_log_file(header_prefix)
log_and_print('Begin Grid Search Run')   
log_and_print('----------------------------------------------------------------') 
#############################################################################
#### Import Config File
#############################################################################
try:
    with open("gs_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
except Exception as e:
        log_and_print(f"Failed to load configuration file. Error: {e}")
        log_and_print(f"Exit Program")
        sys.exit()

#print(config)
#############################################################################
#### Load Data and Parameters
#############################################################################
# Load the dataset
try:
    X_train_df = pd.read_csv('./data/X_train.csv', index_col='student_id')
    y_train_df = pd.read_csv('./data/y_train.csv', index_col='student_id')
except Exception as e:
    log_and_print(f"Failed to load data file. Error: {e}")
    log_and_print(f"Exit Program")
    sys.exit()

#### Get Sample Size
sample_size = config['SAMPLE_SIZE']

#### Set up training data according to the sample size
X_train = X_train_df.sample(frac=sample_size)
y_train = y_train_df.loc[X_train.index]

if sample_size < 1.0:
    log_and_print('----------------------------------------------------------------')
    log_and_print('Training dataset after downsize')
    log_and_print(f'Size of X_train is: {len(X_train)}')
    log_and_print(f'Size of y_train is: {len(y_train)}')


# Store results
results = []
subfolder = 'pkl'

############################################################################
#### Grid Search Using Loop for each model
#############################################################################
models_list = config['MODELS_LIST']
models = config['MODELS']
params_list = config['PARAMS']
grid_params = config['GRID_SEARCH_PARAMS']
#print(grid_params)
pipe_list = yaml_pipe_convert(models_list, models) 

# Loop through each model and its parameter grid
for model_name in models_list:
    log_and_print(f"Running GridSearchCV for {model_name}...")

    # get pipeline
    pipeline = Pipeline(pipe_list.get(model_name))
    
        
    # Set up GridSearchCV with model, parameters, and grid search settings
    try: 
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=params_list[model_name],
            **grid_params)
    except Exception as e:
        log_and_print(f"Failed to set parameters.")
        log_and_print(f"Error: {e}")
        sys.exit()
    
    # Fit GridSearchCV
    try:
        grid_search.fit(X_train, y_train.values.ravel())
    except Exception as e:
        log_and_print(f"Failed to fit grid search")
        log_and_print(f"Error: {e}")
        sys.exit()

    # Save best parameters and best score for each model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_
    results.append({
        'model': model_name,
        'best_score': best_score,
        'best_params': best_params,
        'model_estimator': best_estimator
    })
    
    # Organize grid search result for logging
    gs_result_raw = pd.DataFrame(grid_search.cv_results_)
    gs_result = gs_result_raw[['params','mean_test_score', 'rank_test_score', 'mean_fit_time']] 
    gs_result = gs_result.round({'mean_test_score':4, 'mean_fit_time':1})
    gs_result = gs_result.sort_values('rank_test_score')
    gs_result = gs_result.reset_index(drop=True)
    
    # Logging GS result to the log file
    for idx in range(len(gs_result)):
        score = gs_result.loc[idx, 'mean_test_score']
        params = gs_result.loc[idx, 'params']
        rank = gs_result.loc[idx, 'rank_test_score']
        time = gs_result.loc[idx, 'mean_fit_time']
        logging.info(f"params = {params}: rank = {rank}, score = {score}, time = {time}s")

# Display results and save pkl 
for result in results:
    log_and_print(f"Model: {result['model']}")
    log_and_print(f"Best Score (R²): {result['best_score']:.4f}")
    log_and_print(f"Best Parameters: {result['best_params']}")
    model_name = str(result['model'])
    full_name = 'best_'+model_name+'.pkl'
    pathname = os.path.join(subfolder, full_name)
    joblib.dump(result['model_estimator'], pathname)
    
log_and_print('----------------------------------------------------------------') 
log_and_print('End of Grid Search Run') 