# initial_model_evaluation.py


### Disclaimer ##############################################################
### The original code is generated from ChatGPT                        ######
### However, the code is heavily modified for initial model evaluation ######
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
    log_and_print("xgboost is not installed.")


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
header_prefix = 'init_model_eval'
create_log_file(header_prefix)
log_and_print('Begin Initial Model Evaluation Run') 
log_and_print('----------------------------------------------------------------')   
#############################################################################
#### Import Config File
#############################################################################
try:
    with open("initial_model_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
except Exception as e:
        log_and_print(f"Failed to load configuration file. Error: {e}")
        log_and_print(f"Exit Program")
        sys.exit()



#############################################################################
#### Load Data and Parameters
#############################################################################

try:
    X_train_raw = pd.read_csv('./data/X_train.csv', index_col='student_id')
    y_train_raw = pd.read_csv('./data/y_train.csv', index_col='student_id')
except Exception as e:
    log_and_print(f"Failed to load data file. Error: {e}")
    log_and_print(f"Exit Program")
    sys.exit()

#### Get CV Size
cv_size = config['CV_SIZE']


#### Set up cross validation test data
X_train2, X_cv, y_train2, y_cv = train_test_split(X_train_raw, y_train_raw, test_size=cv_size)
log_and_print('Train and CV dataset formed')
log_and_print(f'Size of X_train is: {len(X_train2)}')
log_and_print(f'Size of y_train is: {len(y_train2)}')
log_and_print(f'Size of X_cv is: {len(X_cv)}')
log_and_print(f'Size of y_cv is: {len(y_cv)}')

#### Get Sample Size
sample_size = config['SAMPLE_SIZE']


#### Set up training data according to the sample size
X_train = X_train2.sample(frac=sample_size)
y_train = y_train2.loc[X_train.index]
log_and_print('----------------------------------------------------------------')
log_and_print('Training dataset after downsize')
log_and_print(f'Size of X_train is: {len(X_train)}')
log_and_print(f'Size of y_train is: {len(y_train)}')


#############################################################################
#### Model Evaluation Using Loop
#############################################################################
models_list = config['MODELS_LIST']
models = config['MODELS']
params_list = config['PARAMS']

pipe_list = yaml_pipe_convert(models_list, models) 
#print(pipe_list)

# Iterate over each model defined in the config
for model_name in models_list:

    # initialization
    best_score = -np.inf
    best_model_name = None
    best_params = None

    # print heading
    log_and_print(' ')
    log_and_print(f"Evaluating model: {model_name}")
    log_and_print('----------------------------------------------------------------')

    # get pipeline
    pipeline = Pipeline(pipe_list.get(model_name))


    # run model with default settings
    try:
        # Fit the pipeline
        pipeline.fit(X_train, y_train.values.ravel())
        
        # Predict and calculate the R² score
        r2 = pipeline.score(X_cv, y_cv)

        log_and_print("Running model with default Settings:")
        log_and_print('----------------------------------------------------------------')
        log_and_print(f'R2 training set: {pipeline.score(X_train, y_train)}')
        log_and_print(f'R2 cross validation set: {pipeline.score(X_cv, y_cv)}')

    except Exception as e:
        log_and_print(f"Failed to fit model {model_name} with parameters {params}:")
        log_and_print(f"Error: {e}")
    
    # Get the hyperparameter grid for the current model
    param_grid = params_list.get(model_name, {})
    
    # Generate all combinations of the hyperparameters
    param_keys = list(param_grid.keys())
    param_values = list(product(*param_grid.values()))

    # Iterate through each combination of parameters
    log_and_print(' ')
    log_and_print('----------------------------------------------------------------')
    log_and_print("Running model with various parameters:")
    log_and_print('----------------------------------------------------------------')
    for param_combination in param_values:
        # Create a dictionary of parameters for this iteration
        params = dict(zip(param_keys, param_combination))
        
        # Set the parameters in the pipeline
        try:
            pipeline.set_params(**params)
        except Exception as e:
            log_and_print(f"Failed to set parameters.")
            log_and_print(f"Error: {e}")
            continue
        
        try:
            # Fit the pipeline
            pipeline.fit(X_train, y_train.values.ravel())
            
            # Predict and calculate the R² score
            r2 = pipeline.score(X_cv, y_cv)

            log_and_print(f"{params}")
            log_and_print(f'R2 training set: {pipeline.score(X_train, y_train)}')
            log_and_print(f'R2 cross validation set: {pipeline.score(X_cv, y_cv)}')
            log_and_print('-----------------------')
            
            # Check if this is the best score so far
            if r2 > best_score:
                best_score = r2
                best_model_name = model_name
                best_params = params

        except Exception as e:
            log_and_print(f"Failed to fit model {model_name} with parameters {params}:")
            log_and_print(f"Error: {e}")
            continue

    # Output the best model and its parameters
    log_and_print('----------------------------------------------------------------')
    log_and_print(f"Best Model: {best_model_name}")
    log_and_print(f"Best R² Score: {best_score:.4f}")
    log_and_print(f"Best Parameters: {best_params}")
    log_and_print('----------------------------------------------------------------')

log_and_print("End of Initial Model Evaluation")