# Hotel No-Show Prediction

## Author
Prepared by Thomas Tay

## Overview
This project **Hotel No-Show Prediction** tries to identify potential customer that may not show up after the booking. The hotel management will for mulate policy to reduce cost based on the prediction.

## Folder Structure
The zip file once expanded will have the following folder structure:
```css
.
├── AIAP Technical Assessment Past Years Series - Hotel No-Show Prediction.pdf
├── README.md
├── eda.ipynb
├── Dockerfile
├── requirements.txt
└── src
    ├── data (default is empty)
    │   ├── X_test.csv
    │   ├── X_train.csv
    │   ├── score.db
    │   ├── y_test.csv
    │   └── y_train.csv
    ├── pkl (default is empty)
    │   ├── best_ridge.pkl
    │   ├── best_decision_tree.pkl
    │   └── ...
    ├── log (default is empty)
    │   ├── data_processing_datetime.log
    │   ├── init_model_eval_datetime.log
    │   ├── gs_model_eval_datetime.log
    │   ├── apply_model_datetime.log
    │   └── ...
    ├── data_processing.py
    ├── initial_model_config.yaml
    ├── initial_model_eval.py
    ├── gs_config.yaml
    ├── gs_models_eval.py
    ├── apply_model_config.yaml
    └── apply_model.py
```
    
- The base folder contains the following files/folder:
    - Project document in PDF format.
    - This README file in markdown format.
    - An EDA file in Jupyter notebook format.
    - A docker file that allows user to build an image from it.
    - A requirement file that contains the installation requirement to be use together when building a container.
    - A src folder that contains the code and configuration file in Python and YAML file format.
-  The src folder contains the following folders and files:
    - **Folders** 
    - In addition to various python file, there are 3 folders that contains the data, pkl and log files.
    - data folder : This folder contains all the data related to this project.
    - pkl folder : This folder contains all the grid search parameters for each model.
    - log folder : For each run of the Python file, there is a log file created and store in the log folder.
    - **Python and Configuration Files**  
    - `data_processing.py` : Use this Python file to download the raw data and perform the necessary data cleaning and feature engineering.
    - `initial_model_config.yaml` : Use this file to perform all the configuration of the hyperparameters for initial model evaluation.
    - `initial_model_eval.py` : Use this file to run initial model evaluation. This is no grid search. Users can select which model to perform initial model evaluation.
    - `gs_config.yaml` : Use this config file to perform hyperparameters fine tuning for grid search.
    - `gs_models_eval.py` : Use this file to perform grid search. For each model, the best parameters are save into the pkl folder.
    - `apply_model_config.yaml` : Use this file to configure which model you want to apply to the datasets. You can also manually configure the parameters.
    - `apply_model.py` : Use this file to apply the best model to the training and test datasets.


## Setup and Installation 
It is recommended to use a container to setup the execution environment. A docker file is provided to build the container.

### Basic Requirements:
- Please have docker installed in the PC or virtual machine.
- The base requirement is Python 3.11. This requirement is specified in the docker file.
- Package requirement also included to work with the docker file.

Package Requirements:
```bash
scikit-learn==1.2.2
numpy==1.23.5
pandas==2.2.2
scipy==1.12.0
xgboost==2.1.2
pyyaml==6.0.2
```

### Step 1: Unzip the Package
Unzip the main zip file into a place where it is convenient to run with a container. Please navigate to the base folder where this README document resides.


### Step 2: Bulding a Container Environment
Before building the container, please ensure that docker is installed on the PC or VM and you are able to build an image from the command prompt.

The Docker file and the requirements file is included in the base folder. You need to build the docker image from the base folder.

To build a container image use the following code, replace the name of image with a name of your liking:
```bash
docker build -t <name-of-image> .
```

### Step 3: Running a Container Image
Please make sure that you are either at the base folder or src folder. To run the docker image, use the following command:
```bash
docker run -it --rm -v $(pwd):/app <name-of-image> 
```

- If you are in the base folder, you need to drill down to src folder to run all the evaluation program.

### Setup Guide for VM or Local Machine
If you are using other VM or local machine without docker, please follow the brief guide below. 

If you do not have any Python installed in your environment, please run the code below:
```bash
apt-get install python3.11 python3.11-venv python3-pip
```

Setting Up and Activate Virtual Environment for Python
```bash
# create a python virtual environment
python3.11 -m venv <path-env-name>
# activate the environment
source <path-env-name>/bin/activate
# To deactivate just type deactivate at the command prompt
```

Install the required Python packages. Please make sure that you navigate to the base folder where the requirement file reside.
```bash
# Navigate to the base folder first.
pip install -r requirements.txt
```

## ML Flow
The general flow of implemeting this project can be found below:
```mermaid
graph TD;
    Data_Processing-->Initial_Model_Evaluation;
    Initial_Model_Evaluation-->Grid_Search;
    Grid_Search-->Apply_Model;
```

The following is a brief description of the ML flow, the execution details are provided under Execution Intructions.

1. First step is downloading and processing data. 
2. The next step is initial model evaluation. As grid search is resource intensive. We can explore all available models and its basic configuration at this stage. We can narrow down our search range before conducting grid search. 
3. The next step is grid search with cross validation fold. Depending on the complexity, you can choose to run multiple models or one model at a time.
4. The final step is to apply the best model on the training and test dataset.

**Note**: For each step, there will be a log file generated. 

**Note**: From step 2 to step 4, there will be a configuration file used to define model, pipeline and parameters.
 
## Instruction for Execution

### Step 1: Data Processing
Please use the following code for data processing, we need to run it from the src folder:

```bash
python data_processing.py
```

**Note** : All processing log will be stored in the log folder with prefix `data_processing` follow by datetime.

### Step 2: Initial Model Evaluation 

The next step is to perform initial model evaluation, we can use this step to perform initial evaluation on various model. As performing grid search is resource intensive. We use this step to look at the basic result of various model. The configuration file `initial_model_config.yaml` allow us to set various parameters, so that we can look at the score of these parameters. As both the training score and cross validation score are printed. We can discover which parameters is overfitting. Please note that at this stage we only use the model score as the evaluation metrics.

#### Configuration Guide
There are 5 main configurations at `initial_model_config.yaml`. They are CV_SIZE, SAMPLE_SIZE, MODEL_LIST, MODELS and PARAMS.

- CV_SIZE : This allow us to set the size of the cross validation dataset.
- SAMPLE_SIZE : If your dataset is very large, you can reduce the size of the training dataset.
- MODEL_LIST : Use this setting to select which model to run. Only the models specfied here will be run.
- MODELS : Use this setting to construct the pipeline for each model.
- PARAMS : Use this setting to define which parameters to run for each model.
  
Please note that this is no grid search, you can set the parameters with a wide range. You can see the training score and test score for each combination. Use this tool to set a large range, look at the score for underfitting and overfitting. Narrow down the search parameters in preparation for grid search.


Setting the size of cross validation dataset. Best recommendation is to set between 0.1 to 0.3 depending on the overall data size. 
```python
# Define the size of cross validation dataset, range from 0.1 to 0.9
CV_SIZE : 0.2
```

If your dataset is very large, you can set the sample size so that we can reduce the data size. Please note that for the existing training dataset, the system will set aside the cross validation dataset first. Then the remaining will be use for training.
```python
# Define Sample size for large dataset
# If your dataset is very large, you can set the sample size
# Please note that the system will extract the CV dataset for testing first
# SAMPLE SIZE will only applied to the remaining training data after the CV Split
# Set from 0.1 to 1.0. 1.0 means not reducing teh dataset size.
SAMPLE_SIZE : 0.6
```

Selecting which model to process. This is a model pick list. We can select which model to evaluate by setting the model list. For larger dataset or complex model, it is best to use only one model in the list.
```python
# Define which model to run
# The MODELS_LIST will define which model to run so there is no need to delete any model configuration
#MODELS_LIST : ['logistic_regression', 'decision_tree_classifier', 'knn_classifier', 'svc', 'rf_classifier', 'gb_classifier', 'xgboost_classifier']
#MODELS_LIST : ['decision_tree_classifier','rf_classifier', 'gb_classifier', 'xgboost_classifier']
MODELS_LIST : ['xgboost_classifier']

```

Configure the pipeline for each model, once the configuration is set, there is little need to change it. However, you can decide which model need polynomial features and feature scaling. You can also decide the type of feature scaling required for each model.  Currently, we only support StandardScaler and MinMaxScaler.
```python
# Define models pipelines that include polynomial features and scaling.
# For feature scaling we have StandardScaler and MinMaxScaler
MODELS : 
  'logistic_regression':
    - 'poly': PolynomialFeatures
    - 'scaler': StandardScaler
    - 'logistic': LogisticRegression 
  'decision_tree_classifier':
    - 'decision_tree_classifier': DecisionTreeClassifier
.........
  'rf_classifier': 
    - 'rf_classifier': RandomForestClassifier
  'gb_classifier': 
    - 'gb_classifier': GradientBoostingClassifier
  'xgboost_classifier': 
    - 'xgboost_classifier': XGBClassifier
........
```

Setting hyperparameters:
We can also set the range of hyperparameters at a large range. The following is just some example:
```python
PARAMS : {
......
   'gb_classifier':{
        'gb_classifier__n_estimators': [50, 100, 200],         # Number of boosting stages
        'gb_classifier__learning_rate': [0.01, 0.1, 0.2],      # Shrinks contribution of each tree
        'gb_classifier__max_depth': [3, 5, 10],                # Maximum depth of the tree
        'gb_classifier__min_samples_split': [2, 5, 10],        # Minimum samples required to split a node
        'gb_classifier__min_samples_leaf': [1, 2, 4],          # Minimum samples required at each leaf node
        'gb_classifier__subsample': [0.8, 0.9, 1.0],           # Fraction of samples used for fitting each tree
        'gb_classifier__max_features': ['sqrt', 'log2', None], # Number of features to consider for splits      
    },
......
```

#### Execution Command
Please use the following code to run initial model evaluation, we need to run it from the src folder:
```bash
python initial_model_eval.py
```

**Note** : All processing log will be stored in the log folder with prefix `init_model_eval` follow by datetime.

### Step 3: Grid Search and Model Evaluation

This step allow us to perform fine tuning and perform grid search on the more promising models. We can also set the evaluation matrix at the `gs_config.yaml` file. Similarly, we can also set sample size, and model list to choose which model to use for evaluation. Please note that in grid search, the best parameters for each model will be saved into a pkl file under the pkl folder.

#### Configuration Guide
There are 5 main configurations. They are SAMPLE_SIZE, MODEL_LIST, MODELS, PARAMS and GRID_SEARCH_PARAMS
- SAMPLE_SIZE : If you dataset is very large, you can reduce the taining dataset.
- MODEL_LIST : Use this setting to select which model to run, this way we can keep the settings of all other models.
- MODELS : Use this setting to construct the pipeline for each model.
- PARAMS : Use this setting to define which parameters to run for each model
- GRID_SEARCH_PARAMS : This is the settings for grid search function

If your dataset is very large, you can set the sample size so that we can reduce the data size. Use 1.0 to make use of the entire training dataset.
```python
# Define Sample size for large dataset
# If your dataset is very large, you can set the sample size
# Use 1.0 for the whole dataset
SAMPLE_SIZE : 1.0
```

Select which model to process. This is a model pick list. For larger dataset or complex model, it is best to use only one model in the list.
```python
# Define which model to run
# The MODELS_LIST will define which model to run so there is no need to delete any model configuration
#MODELS_LIST : ['logistic_regression', 'decision_tree_classifier', 'knn_classifier', 'svc', 'rf_classifier', 'gb_classifier', 'xgboost_classifier']
#MODELS_LIST : ['decision_tree_classifier','rf_classifier', 'gb_classifier', 'xgboost_classifier']
MODELS_LIST : ['xgboost_classifier']
```

Configure the pipeline for each model, you can decide the type of feature scaling required for each model.  Currently, we only support StandardScaler and MinMaxScaler.
```python
# Define models pipelines that include polynomial features and scaling
MODELS = {
......
   'decision_tree_classifier':
    - 'decision_tree_classifier': DecisionTreeClassifier
  'knn_classifier':
    - 'poly': PolynomialFeatures
    - 'scaler': StandardScaler
    - 'knn_classifier': KNeighborsClassifier        
  'svc':
    - 'poly': PolynomialFeatures
    - 'scaler': StandardScaler
    - 'svc': SVC       
  'rf_classifier': 
    - 'rf_classifier': RandomForestClassifier     
........
}
```


Setting hyperparameters:
We can also set the range of hyperparameters for grid search. The following is just one example:
```python
PARAMS : {
   'logistic_regression': {
        'poly__degree': [2], 
        'logistic__C': [0.005, 0.008],
        'logistic__penalty': ['l2'], 
        'logistic__max_iter': [10000],
        'logistic__solver': ['lbfgs', 'liblinear']           
    },
    'decision_tree_classifier': {
        'decision_tree_classifier__criterion': ['gini'],
        'decision_tree_classifier__max_depth': [5, 10, 50],
        'decision_tree_classifier__min_samples_split': [2, 3, 5],
        'decision_tree_classifier__min_samples_leaf': [1, 3, 4],
        'decision_tree_classifier__max_features': [0.4, 0.5],
        'decision_tree_classifier__max_leaf_nodes': [50, 100, 200]
    },
    'knn_classifier': {
        'poly__degree': [2],
        'knn_classifier__n_neighbors': [15, 20, 25],
        'knn_classifier__weights': ['distance'],
        'knn_classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn_classifier__p': [1, 2]
    },
...
```

We can also set grid search parameters and evaluation metrics at:
```python
GRID_SEARCH_PARAMS : {
    'cv': 5,                        # Number of cross-validation folds
    'scoring': 'accuracy',          # Use Accuracy as the scoring metric
    'n_jobs': 4,                    # Do not use -1 as it will cause issue 
    'verbose': 3
}
```

#### Execution Command
Please use the following code to run grid search, we need to run it from the src folder:
```bash
python gs_models_eval.py
```

**Note** : All processing log will be stored in the log folder with prefix `gs_model_eval` follow by datetime.

### Step 5: Apply Model
This step allow us to to apply the best model from grid search. We can choose to use which model and there is an option to configure the model manually. We can select the models and set the parameters at `apply_model_config.yaml` file. This is also the first time we use the test data for the final evaluation. 


#### Configuration Guide
There are 5 main configurations. They are BEST_MODELS_PKL, BEST_MODELS_LIST, MODELS and PARAMS
- BEST_MODELS_PKL : This is where we choose the best model and use the pkl file stored in the pkl folder.
- BEST_MODELS_LIST : We use this list to decide which model we want to customized
- MODELS : Use this setting to construct the pipeline for each model.
- PARAMS : Use this setting to define which parameters to run for each model. This is no grid search. So please use one setting for each parameter.

**Note**: This will be the first time we make use of the test dataset. 

Selecting which model to apply to the training and test dataset. This is a model pick list. The system will use the pkl file stored in the pkl model. 
```python
# Enter the model to use pkl file
#BEST_MODELS_PKL : []
#BEST_MODELS_PKL : ['logistic_regression', 'decision_tree_classifier', 'knn_classifier', 'svc', 'rf_classifier', 'gb_classifier', 'xgboost_classifier']
BEST_MODELS_PKL : ['decision_tree_classifier']
```

Selecting which model to apply to the training and test dataset. This is a model pick list. The system will NOT use the pkl file. It will use the manual configuration specified below.
```python
# Enter the model for manual configuration
#BEST_MODELS_LIST : []
#BEST_MODELS_LIST : ['logistic_regression', 'decision_tree_classifier', 'knn_classifier', 'svc', 'rf_classifier', 'gb_classifier', 'xgboost_classifier']
#BEST_MODELS_LIST : ['decision_tree_classifier','rf_classifier', 'gb_classifier', 'xgboost_classifier']
BEST_MODELS_LIST : ['logistic_regression', 'decision_tree_classifier']
```

Configure the pipeline for each model, once the configuration is set there is little need to change it. We only support StandardScaler and MinMaxScaler.
```python
# Define models pipelines that include polynomial features and scaling.
# For feature scaling we have StandardScaler and MinMaxScaler
MODELS : 
  'logistic_regression':
    - 'poly': PolynomialFeatures
    - 'scaler': StandardScaler
    - 'logistic': LogisticRegression 
  'decision_tree_classifier':
    - 'decision_tree_classifier': DecisionTreeClassifier
  'knn_classifier':
    - 'poly': PolynomialFeatures
    - 'scaler': StandardScaler
    - 'knn_classifier': KNeighborsClassifier        
  'svc':
    - 'poly': PolynomialFeatures
    - 'scaler': StandardScaler
    - 'svc': SVC       
  'rf_classifier': 
    - 'rf_classifier': RandomForestClassifier
......
```


Setting hyperparameters:
Please note that this is no grid search, use only one setting per parameter.
```python
# Define hyperparameters for each model, including polynomial degrees
# THIS IS NOT GRID SEARCH, USE ONE SETTINGS ONLY
PARAMS : {
    'logistic_regression': {
        'poly__degree': 2, 
        'logistic__C': 0.01,
        'logistic__penalty': 'l2', 
        'logistic__max_iter': 10000,
        'logistic__solver': 'lbfgs',             
    },
    'decision_tree_classifier': {
        'decision_tree_classifier__criterion': 'gini',
        'decision_tree_classifier__max_depth': 20,
        'decision_tree_classifier__min_samples_split': 5,
        'decision_tree_classifier__min_samples_leaf': 1,
        'decision_tree_classifier__max_features': 'sqrt',
        'decision_tree_classifier__max_leaf_nodes': 20,
    },
......
```


Please use the following code to apply the model, we need to run it from the src folder:
```bash
python apply_model.py
```


## Key Findings in EDA
The EDA was conducted using a Jupyter Notebook (`notebooks/eda.ipynb`). Key insights include:

- Data Index: The column `booking_id` is the key identifier and should be use as index.
- Null Record: There is only one record with all the null values. This record is deleted.
- Missing Feature Values: There are many null values for the column `room` and `local_price`. For missing room type, we fill the missing data with most frequent room type in the dataset which is 'King`.
- For missing prices, we take the average price of same room type on the same day and at the same branch. If there is no values to compute the average pricing, we will take the average pricing of the same room, on the same month and at the same branch. Last resort, we use the same room type on the same month.
- Correlated Features: Some features have near zero correlation, such as `platform`, `booking_month`, `arrival_month` and `arrival_day`.


## Feature Engineering Decisions:
- Create a new column `local_price` and convert the prices in USD to SGD.
- Feature Scaling: Applied standardization only to models that required scaling. Decision-tree based model will have no scaling.
- Categorical Encoding: Used one-hot encoding for categorical variables such as `branch`, `country`, `first_time`, `room` and `platform`.
- We use custom mapping for all months related columns such as `booking_month`, `arrival_month` and `checkout_month`. The scheme is such that January is 1 and December is 12.
- New Features: Create a new column `days_stayed` to check if the duration of stay has any impact on the target.
- New Features: Create a new column `mths_adv_booking` to calculate the number of months in advance booking.

## Data Processing and Feature Engineering Summary Table
| Feature Names | Type           | Processing Steps  |
| ------------- |:-------------:| :-----|
| booking_id     | str | This is unique, and set as index |
| local_price     | float | This is created by converting the USD prices to local prices. Also remove column price. For missing value, use average price of same day, room type and location. |
| days_stayed | int | Use the difference between arrival date and checkout date |
| mths_adv_booking | int | Compute the difference between the booking month and arrival month. If booking months is later, then we assume it is the month from previous year. |
| room | str | For room with missing values, we use the most commonly appear room type. |


## Evaluation Metrics
The default evaluation metric for classification problem is accuracy score. This metric measure how accuracy is the prediction.


## Initial Model Selection and Evaluation
The strategy is to select all models and set the parameters at the very large range. Look at the output and narrow down the gap between parameters.
We can also use the result to drop models that are not performing well.

### Initial Results

**Logistics Regression**
<img width="1247" alt="Screenshot 2024-12-03 at 17 01 01" src="https://github.com/user-attachments/assets/cff19a4e-a369-42e6-8737-8b570840dce4">

**Decision Tree Classifier**
<img width="1140" alt="Screenshot 2024-12-03 at 17 02 29" src="https://github.com/user-attachments/assets/aa538165-b6e2-44e8-bdd8-a30a5e68104b">

**KNN Classifier**
<img width="1173" alt="Screenshot 2024-12-03 at 17 03 37" src="https://github.com/user-attachments/assets/7d8500c6-4e99-41c8-9d81-313040519b1a">

**Random Forest Classifiers**
<img width="1302" alt="Screenshot 2024-12-03 at 17 04 10" src="https://github.com/user-attachments/assets/d7409bf4-7f52-43fc-8004-07fc7691fd18">

**Gradient Boosting Classifier**
<img width="1321" alt="Screenshot 2024-12-03 at 17 05 47" src="https://github.com/user-attachments/assets/075ffb69-d7e2-4eb0-8603-ef35a4ce0433">

**XGBoost Classifier**
<img width="1325" alt="Screenshot 2024-12-03 at 17 04 54" src="https://github.com/user-attachments/assets/6bd5d641-a72c-41b5-a859-19b180bf9299">

Based on the initial result, logistic regression will be the based model.

The model to search will be decision tree model and its variant such as Random Forest, Gradient Boost and XGBoost.

## Grid Search and Model Evaluation
**Baseline Model: Logistics Regression**
```python
'logistic_regression': {
        'poly__degree': [2], 
        'logistic__C': [0.005, 0.008],
        'logistic__penalty': ['l2'], 
        'logistic__max_iter': [10000],
        'logistic__solver': ['lbfgs', 'liblinear']           
    },
```

<img width="1288" alt="Screenshot 2024-12-03 at 17 45 59" src="https://github.com/user-attachments/assets/cc1f6cf6-a97d-476e-838c-6d30c5b274be">


**Decision Tree Classifier**
```python
'decision_tree_classifier': {
        'decision_tree_classifier__criterion': ['gini'],
        'decision_tree_classifier__max_depth': [5, 10, 50],
        'decision_tree_classifier__min_samples_split': [2, 3, 5],
        'decision_tree_classifier__min_samples_leaf': [1, 3, 4],
        'decision_tree_classifier__max_features': [0.4, 0.5],
        'decision_tree_classifier__max_leaf_nodes': [50, 100, 200]
    },
```
<img width="1194" alt="Screenshot 2024-12-03 at 17 46 10" src="https://github.com/user-attachments/assets/c10f351f-32d8-4770-a9b1-a36e0208407e">



**Random Forest Classifier**
```python
'rf_classifier':{
        'rf_classifier__n_estimators': [500, 600],   
        'rf_classifier__max_depth': [30, 35, 40],     
        'rf_classifier__min_samples_split': [2, 3, 5],          
        'rf_classifier__min_samples_leaf': [1, 2, 4],           
        'rf_classifier__max_features': ['sqrt', 'log2'],        
        'rf_classifier__class_weight': ['balanced_subsample', 'balanced'],  
    },
```




**Gradient Boosting Classifier**
```python
'gb_classifier':{
        'gb_classifier__n_estimators': [300, 500, 800],       
        'gb_classifier__learning_rate': [0.005,0.01, 0.05],     
        'gb_classifier__max_depth': [5, 10, 20],              
        'gb_classifier__min_samples_split': [5, 10, 15, 20],     
        'gb_classifier__min_samples_leaf': [1, 2, 4, 10],          
        'gb_classifier__subsample': [0.5, 0.9, 1.0],               
        'gb_classifier__max_features': ['sqrt', 'log2'],                 
    },

```



**XGBoost Classifier**
```python
'xgboost_classifier': {
        'xgboost_classifier__n_estimators': [150, 200, 250, 300],           
        'xgboost_classifier__learning_rate': [0.005, 0.01, 0.05],        
        'xgboost_classifier__max_depth': [15, 20, 25],                            
        'xgboost_classifier__gamma': [0.1],                   
        'xgboost_classifier__subsample': [0.8, 0.9, 1.0],                     
    }
```
<img width="1298" alt="Screenshot 2024-12-03 at 17 48 00" src="https://github.com/user-attachments/assets/17a0c5c4-1c62-4c3d-bdee-520167047edd">




## Model Evaluation

Base on the grid search result above, we will run the `apply_model.py` program to apply the the best model model on the test data. We can use the pkl file from the grid search or alternatively, to do that, we can use `apply_model_config.py` file to select which pkl file to use. Additionally, we can also manually configure our desired parameters in the same config file.

The following is the result:
```

-----------------------------
```

Base on the result above, we recommend Gradient Boosting Regressor or XGBoost as our preferred model as it generate the best R2 score on the test datasets.

