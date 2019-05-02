# CAPP 30254 Machine Learning for Public Policy
# Homework 3 - Improving the Pipeline
#
# Pipeline Configuration file
# Description: This file holds all hard-coded values for the HW3 ML pipeline,
#   including file paths, model parameters, etc. The section headers correspond
#   to the specific portion of the assignment where the particular config
#   variable is used.

################
# 1. READ DATA #
################

# Filepath where credit card data is stored
DATA_PATH = 'data/projects_2012_2013.csv'

#######################
# 5. BUILD CLASSIFIER #
#######################

# Identifying column of interest
LABEL = 'fully_funded_in_60_days'

# Proportion of full data to use as a test set
TEST_SIZE = 0.3

# Probability threshold for classifying an observation as positive
CLASS_THRESHOLD = 0.7

# Nested dictionary of model parameters
MODEL_PARAMS = {
    'LogisticRegression': {
        'penalty': 'l1',
        'solver': 'liblinear',
        'random_state': 0
    },
    'KNeighborsClassifier': {
        'weights': 'distance'
    },
    'DecisionTreeClassifier': {
        'criterion': 'entropy',
        'max_features': 'auto',
        'random_state': 0
    },
    'LinearSVC': {
        'penalty': 'l2',
        'C': 1.0,
        'random_state': 0
    },
    'RandomForestClassifier': {
        'criterion': 'entropy',
        'max_features': 'auto',
        'random_state': 0
    },
    'AdaBoostClassifier': {
        'random_state': 0
    },
    'BaggingClassifier': {
        'random_state': 0
    }
}
