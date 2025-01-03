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
#### Price Entry
###############################################################
SGD_one_USD = 1.35

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

url = 'https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db'

file_path = Path('./data/noshow.db')

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
conn = sqlite3.connect('./data/noshow.db')
df = pd.read_sql(read_table, conn)
conn.close()
log_and_print('Data Extracted Close Connection')

###############################################################
#### Data Cleaning Before Data Split
###############################################################
log_and_print('Starting the first part of data cleaning')
log_and_print('----------------------------------------------------------------')
log_and_print('Set booking_id as index')
df.set_index('booking_id', inplace=True)
log_and_print('Remove record with all null values:')
df = df.drop(df[df['no_show'].isnull()].index)
log_and_print(f"There are a total of {len(df)} records.")
log_and_print('Fix col: num_adults')
df.num_adults = df.num_adults.replace('one', '1')
df.num_adults = df.num_adults.replace('two', '2')
df.num_adults = df.num_adults.astype(int)
log_and_print('Fix col: no_show and num_children')
df.num_children = df.num_children.astype(int)
df.no_show = df.no_show.astype(int)

#### Price Processing
log_and_print('Starting price processing')
log_and_print('----------------------------------------------------------------')
price_df = df['price'].str.split(' ', expand = True).copy()
price_df.columns = ['CUR', 'price']
price_df.price = price_df.price.astype(float)
price_df['local_price'] = price_df['price'].where(price_df.CUR != 'USD$', price_df.price * SGD_one_USD)
df2 = pd.merge(df, price_df['local_price'], left_index=True, right_index=True)
df2.drop(columns=['price'], inplace=True)
log_and_print('Price converted with column name local_price')

#### Fixing arrival_month and checkout_day Issue¶
log_and_print('Fixing arrival_month and checkout_day issue')
log_and_print('----------------------------------------------------------------')
df2['arrival_month'] = df2['arrival_month'].str.capitalize()
df2['checkout_day'] = abs(df2['checkout_day']).astype(int)
df2['arrival_day'] = df2['arrival_day'].astype(int)


###############################################################
#### Feature Engineering
###############################################################

#### Feature Engineering : Duration of Stay
log_and_print('Feature Engineering : Duration of Stay')
log_and_print('----------------------------------------------------------------')
date_df = df2[['booking_month', 'arrival_month', 'arrival_day', 'checkout_month', 'checkout_day']].copy()
date_df['year']= '2020'
date_df['start_date'] = pd.to_datetime(date_df['arrival_day'].astype(str)+'-'+date_df['arrival_month']+'-'+date_df['year'], format='%d-%B-%Y')
date_df['end_date'] = pd.to_datetime(date_df['checkout_day'].astype(str)+'-'+date_df['checkout_month']+'-'+date_df['year'], format='%d-%B-%Y')
date_df['end_date'] = date_df['end_date'].where(date_df['start_date'] < date_df['end_date'], date_df['end_date'] + pd.DateOffset(years=1))
date_df['days_stayed'] = (date_df['end_date'] - date_df['start_date']).dt.days
df3 = pd.merge(df2, date_df['days_stayed'], left_index=True, right_index=True)
log_and_print("New column 'days_stayed' created and added to the dataframe.")

#### Feature Engineering: Number of Months of Advance Booking
log_and_print('Feature Engineering: Number of Months of Advance Booking')
log_and_print('----------------------------------------------------------------')
date_df['mths_adv_booking'] = date_df['start_date'].dt.month - pd.to_datetime(date_df['booking_month'], format='%B').dt.month
date_df['mths_adv_booking'] = date_df['mths_adv_booking'].where(date_df['mths_adv_booking'] >= 0, date_df['mths_adv_booking']+12)
df4 = pd.merge(df3, date_df['mths_adv_booking'], left_index=True, right_index=True)
log_and_print("New column mths_adv_booking created and added to the dataframe.")
log_and_print('----------------------------------------------------------------')
log_and_print('First part of data cleaning completed')
log_and_print('----------------------------------------------------------------')


###############################################################
#### Data Spliting
###############################################################
log_and_print('Forming training and testing dataset')
log_and_print('----------------------------------------------------------------')
y = df4['no_show']
X = df4.drop(columns=['no_show'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
log_and_print('Train and test dataset formed')
log_and_print(f'Size of X_train is: {len(X_train)}')
log_and_print(f'Size of y_train is: {len(y_train)}')
log_and_print(f'Size of X_test is: {len(X_test)}')
log_and_print(f'Size of y_test is: {len(y_test)}')
log_and_print('After the spliting of training and test data. All following data processing will be done on X_train and X_test.')
log_and_print('----------------------------------------------------------------')

###############################################################
#### Data Cleaning After Data Split
###############################################################
log_and_print('Staring part 2 of data cleaning which will be done on X_train and X_test.')
log_and_print('----------------------------------------------------------------')
log_and_print("Fixing Null Values on Column 'room'")
X_train['room'] = X_train['room'].fillna(X_train.room.mode()[0])
X_test['room'] = X_test['room'].fillna(X_test.room.mode()[0])
log_and_print("Null values are filled with room.mode")

#### Feature Engineering: Number of Months of Advance Booking
log_and_print("Fixing Null Values on Column 'local_price'")
log_and_print('----------------------------------------------------------------')
X_train['local_price'] = X_train.groupby(['room', 'arrival_month', 'arrival_day', 'branch'], observed=False)['local_price'].transform(lambda x: x.fillna(x.mean()))
X_train['local_price'] = X_train.groupby(['room', 'arrival_month', 'branch'], observed=False)['local_price'].transform(lambda x: x.fillna(x.mean()))
X_train['local_price'] = X_train.groupby(['room', 'arrival_month'], observed=False)['local_price'].transform(lambda x: x.fillna(x.mean()))
log_and_print(f"Null Values Check for X_train\n")
log_and_print(f"{X_train.isnull().sum()}")
X_test['local_price'] = X_test.groupby(['room', 'arrival_month', 'arrival_day', 'branch'], observed=False)['local_price'].transform(lambda x: x.fillna(x.mean()))
X_test['local_price'] = X_test.groupby(['room', 'arrival_month', 'branch'], observed=False)['local_price'].transform(lambda x: x.fillna(x.mean()))
X_test['local_price'] = X_test.groupby(['room', 'arrival_month'], observed=False)['local_price'].transform(lambda x: x.fillna(x.mean()))
log_and_print(f"Null Values Check for X_test\n")
log_and_print(f"{X_test.isnull().sum()}")
log_and_print('----------------------------------------------------------------')

###############################################################
#### Feature Engineering : Date Custom Mapping
###############################################################
log_and_print("Date Custom Mapping")
log_and_print('----------------------------------------------------------------')
month_order = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
X_train['booking_month'] = X_train['booking_month'].map(month_order)
X_train['arrival_month'] = X_train['arrival_month'].map(month_order)
X_train['checkout_month'] = X_train['checkout_month'].map(month_order)

X_test['booking_month'] = X_test['booking_month'].map(month_order)
X_test['arrival_month'] = X_test['arrival_month'].map(month_order)
X_test['checkout_month'] = X_test['checkout_month'].map(month_order)

log_and_print('Months mapping completed')
log_and_print('----------------------------------------------------------------')

###############################################################
#### Feature Engineering : One Hot Encoding
###############################################################
log_and_print("One-Hot Encoding")
log_and_print('----------------------------------------------------------------')
X_train = pd.get_dummies(X_train, columns=['branch', 'country', 'first_time', 'room', 'platform'], prefix=['br','cty_fr','1st_tm', 'rm', 'pltfm'])
X_test = pd.get_dummies(X_test, columns=['branch', 'country', 'first_time', 'room', 'platform'], prefix=['br','cty_fr','1st_tm', 'rm', 'pltfm'])
log_and_print(f"X_train: Total number of rows = {len(X_train)}: Total number of cololumns = {X_train.shape[1]} ")
log_and_print(f"y_train: Total number of rows = {len(y_train)}")
log_and_print(f"X_test: Total number of rows = {len(X_test)}: Total number of cololumns = {X_test.shape[1]} ")
log_and_print(f"y_test: Total number of rows = {len(y_test)}")
log_and_print('----------------------------------------------------------------')
log_and_print(f"Converting 'branch' with prefix 'br_'.")
log_and_print(f"Converting 'country' with prefix 'cty_fr_'.")
log_and_print(f"Converting 'first_time' with prefix '1st_tm_'.")
log_and_print(f"Converting 'room' with prefix 'rm_'.")
log_and_print(f"Converting 'platform' with prefix 'pltfm_'.")
log_and_print('----------------------------------------------------------------')
log_and_print('Display Screen Info')
X_train.info()
X_test.info()

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