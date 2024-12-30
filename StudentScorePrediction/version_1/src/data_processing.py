# data_processing.py

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


# SQL Component
from urllib.request import urlretrieve
import sqlite3

## SciKit Learning Preprocessing  
from sklearn import preprocessing

## SciKit Learn Train Test Split
from sklearn.model_selection import train_test_split


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

## Custom Helper Function
from myHelper import create_log_file, log_and_print, check_make_folder


# Check folder
check_make_folder()

###############################################################
#### Start Logging
###############################################################
# Set logging prefix
header_prefix = 'data_processing'
create_log_file(header_prefix)
log_and_print('Begin Data Processing Run')
log_and_print('----------------------------------------------------------------')

###############################################################
#### Data Download
###############################################################

url = 'https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db'

file_path = Path('./data/score.db')

if file_path.exists():
    log_and_print(f'File already exist, skip download.')
else:
    try:
        log_and_print('Downloading Data File')
        urlretrieve(url, file_path)
        log_and_print('Download completed')
    except:
        log_and_print(f"Error downloading file. Please check if the file exist at the location: {url}")
        log_and_print(f"Please also check if Internet connection is present.")



###############################################################
#### Connect and Open Database
###############################################################
list_all_tables = "SELECT name FROM sqlite_master WHERE type='table';"

conn = sqlite3.connect(file_path)
cur = conn.cursor()
cur.execute(list_all_tables)
tables_all = cur.fetchall()[0]
tables_all = list(tables_all)
log_and_print(f'All tables in this db: {tables_all}')
conn.close()

read_table = 'SELECT * FROM ' + tables_all[0]
conn = sqlite3.connect('./data/score.db')
df = pd.read_sql(read_table, conn)
conn.close()
log_and_print('Data Extracted Close Connection')

###############################################################
#### Data Cleaning Before Data Split
###############################################################
log_and_print('Starting the first part of data cleaning')
log_and_print('----------------------------------------------------------------')
df.set_index('index', inplace=True)
log_and_print('Removing Duplicates')
df2 = df.drop_duplicates('student_id')
log_and_print('Set student_id as index')
df2.set_index('student_id', inplace=True)
log_and_print('Fixing null values for target')
df2 = df2.drop(df2[df2['final_test'].isnull()].index)
log_and_print('First part of data cleaning completed')

###############################################################
#### Data Spliting
###############################################################
log_and_print('Forming training and testing dataset')
log_and_print('----------------------------------------------------------------')
y = df2['final_test']
X = df2.drop(columns=['final_test'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
log_and_print('Train and test dataset formed')
log_and_print(f'Size of X_train is: {len(X_train)}')
log_and_print(f'Size of y_train is: {len(y_train)}')
log_and_print(f'Size of X_test is: {len(X_test)}')
log_and_print(f'Size of y_test is: {len(y_test)}')
log_and_print('After the spliting of training and test data. All following data processing will be done on X_train and X_test.')

###############################################################
#### Data Cleaning After Data Split
###############################################################
log_and_print('Staring part 2 of data cleaning which will be done on X_train and X_test.')
log_and_print('----------------------------------------------------------------')
log_and_print("Fixing Null Values on Column 'attendance_rate'")
X_train['attendance_rate'] = X_train['attendance_rate'].fillna(X_train['attendance_rate'].median())
X_test['attendance_rate'] = X_test['attendance_rate'].fillna(X_test['attendance_rate'].median())



###############################################################
#### Feature Engineering
###############################################################
log_and_print("Feature Engineering")
log_and_print("Create new column for sleep hours")
X_train['sleep_time'] = pd.to_datetime(X_train['sleep_time'], format='%H:%M')
X_train['wake_time'] = pd.to_datetime(X_train['wake_time'], format='%H:%M')
X_train['sleep_hours'] = (X_train['wake_time']- X_train['sleep_time']).dt.components.hours
X_train.drop(columns=['sleep_time', 'wake_time'], inplace=True)

X_test['sleep_time'] = pd.to_datetime(X_test['sleep_time'], format='%H:%M')
X_test['wake_time'] = pd.to_datetime(X_test['wake_time'], format='%H:%M')
X_test['sleep_hours'] = (X_test['wake_time']- X_test['sleep_time']).dt.components.hours
X_test.drop(columns=['sleep_time', 'wake_time'], inplace=True)

log_and_print("Fixing Categorical Duplication")
X_train['CCA'] = X_train['CCA'].str.capitalize()
X_train['tuition'] = X_train['tuition'].replace('Y', value='Yes')
X_train['tuition'] = X_train['tuition'].replace('N', value='No')

X_test['CCA'] = X_test['CCA'].str.capitalize()
X_test['tuition'] = X_test['tuition'].replace('Y', value='Yes')
X_test['tuition'] = X_test['tuition'].replace('N', value='No')

###############################################################
#### Feature Engineering : One Hot Encoding
###############################################################
log_and_print("One-Hot Encoding")
X_train = pd.get_dummies(X_train, columns=['direct_admission', 'CCA', 'learning_style', 'gender', 'tuition', 'bag_color', 'mode_of_transport'], prefix=['admission','CCA','learnStyle', 'gen', 'tuition', 'bag_color', 'mode_of_transport'])
X_test = pd.get_dummies(X_test, columns=['direct_admission', 'CCA', 'learning_style', 'gender', 'tuition', 'bag_color', 'mode_of_transport'], prefix=['admission','CCA','learnStyle', 'gen', 'tuition', 'bag_color', 'mode_of_transport'])

log_and_print(f"X_train: Total number of rows = {len(X_train)}: Total number of cololumns = {X_train.shape[1]} ")
log_and_print(f"y_train: Total number of rows = {len(y_train)}")
log_and_print(f"X_test: Total number of rows = {len(X_test)}: Total number of cololumns = {X_test.shape[1]} ")
log_and_print(f"y_test: Total number of rows = {len(y_test)}")

log_and_print('----------------------------------------------------------------')
log_and_print("Saving Data Files")
try:
    X_train.to_csv("./data/X_train.csv")
    X_test.to_csv("./data/X_test.csv")
    y_train.to_csv("./data/y_train.csv")
    y_test.to_csv("./data/y_test.csv")
    log_and_print(f"Save complete")
except Exception as e:
    log_and_print(f"Error saving file. Error: {e}")
    log_and_print(f"Make sure a subfolder is named data is created and run this process again.")

log_and_print('----------------------------------------------------------------')
log_and_print('End Data Processing Run')