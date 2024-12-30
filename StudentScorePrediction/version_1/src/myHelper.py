# myHelper.py

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
from pathlib import Path
import itertools 
from itertools import product
import joblib
import yaml
from yaml import Loader
import logging


#############################################################################
#### Folder Check Function
#############################################################################
def check_make_folder():
    data_path = Path('./data')
    log_path = Path('./log')
    pkl_path = Path('./pkl')
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f'Data folder not exists. Create folder')

    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print(f'Log folder not exists. Create folder')

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
        print(f'PKL folder not exists. Create folder')

#############################################################################
#### Logging Helper Functions
#############################################################################

# The following is created by Gemini with some modefication
def create_log_file(header_prefix):
    """
    header_prefix is string based argument
    Creates a log file with a unique timestamped filename.
    
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"./log/{header_prefix}_{timestamp}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(message)s')

def log_and_print(message):
    """Logs a message to a file and prints it to the console."""
    print(message)
    logging.info(message)


#############################################################################
#### YAML Helper Functions
#############################################################################
def yaml_pipe_convert(model_list, models):
    component_mapping = {
        "PolynomialFeatures": PolynomialFeatures,
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "KNeighborsRegressor": KNeighborsRegressor,
        "SVR": SVR,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "XGBRegressor": XGBRegressor
    }

    pipe_list = {}
    for model in model_list:
        model_pipe  = models[model]
        #print('model:', model)
        #print(model_pipe)
        step_list = []
        for eachStep in model_pipe:
            #print(eachStep)
            for step_name, step_func in eachStep.items():
                #print(step_name)
                #print(step_func)
                step_obj = component_mapping[step_func]()
                step_list.append((step_name, step_obj))
        #print(step_list)
        pipe_list[model] = step_list
    return pipe_list