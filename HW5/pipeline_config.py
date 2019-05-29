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

########################
# 3. TEST/TRAIN SPLITS #
########################

# Proportion of full data to use as a test set, if not using temporal splits
TEST_SIZE = 0.3

# Dates for temporal splits
TEMPORAL_SPLITS = [
    {
        'test_start': '7/1/2012',
        'test_end': '12/31/2012'
    },
    {
        'test_start': '1/1/2013',
        'test_end': '6/30/2013'
    },
    {
        'test_start': '7/1/2013',
        'test_end': '12/31/2013'
    },
]

#######################
# 5. BUILD CLASSIFIER #
#######################

# Identifying column of interest
LABEL = 'not_funded_60_days'

# Large grid - most exhaustive option
GRID_LARGE = {
    'LogisticRegression': [
        {'penalty': x, 'C': y, 'solver': 'liblinear', 'random_state': 0} \
        for x in ('l1', 'l2') \
        for y in (0.01, 0.1, 1, 10, 100) \
    ],
    'KNeighborsClassifier': [
        {'n_neighbors': x, 'weights': y, 'algorithm': z} \
        for x in (1, 5, 10, 20, 50) \
        for y in ('uniform', 'distance') \
        for z in ('auto', 'ball_tree', 'kd_tree')
    ],
    'DecisionTreeClassifier': [
        {'max_depth': x, 'max_features': y, 'min_samples_leaf': z,
        'random_state': 0} \
        for x in (1, 5, 10, 50) \
        for y in ('sqrt', 'log2', None) \
        for z in (1, 5, 10)
    ],
    'LinearSVC': [
        {'penalty': x, 'C': y, 'random_state': 0} \
        for x in ('l1', 'l2') \
        for y in (0.01, 0.1, 1, 10, 100)
    ],
    'RandomForestClassifier': [
        {'n_estimators': w, 'max_depth': x, 'max_features': y,
        'min_samples_leaf': z, 'random_state': 0} \
        for w in (10, 100, 1000) \
        for x in (1, 5, 10, 50) \
        for y in ('sqrt', 'log2', None) \
        for z in (1, 5, 10)
    ],
    'AdaBoostClassifier': [
        {'n_estimators': x, 'algorithm': y, 'random_state': 0} \
        for x in (10, 100, 1000) \
        for y in ('SAMME', 'SAMME.R')
    ],
    'BaggingClassifier': [
        {'n_estimators': x, 'random_state': 0} \
        for x in (10, 100, 1000)
    ]
}

# Test grid to make sure everything works - 1 model per classifier
GRID_TEST = {
    'LogisticRegression': [
        {'penalty': 'l2', 'C': 1, 'solver': 'liblinear', 'random_state': 0}
    ],
    'KNeighborsClassifier': [
        {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto'}
    ],
    'DecisionTreeClassifier': [
        {'max_depth': 1, 'max_features': 'sqrt', 'min_samples_leaf': 1,
        'random_state': 0}
    ],
    'LinearSVC': [
        {'penalty': 'l2', 'C': 1, 'random_state': 0}
    ],
    'RandomForestClassifier': [
        {'n_estimators': 10, 'max_depth': 1, 'max_features': 10,
        'min_samples_leaf': 10, 'random_state': 0}
    ],
    'AdaBoostClassifier': [
        {'n_estimators': 10, 'algorithm': 'SAMME.R', 'random_state': 0}
    ],
    'BaggingClassifier': [
        {'n_estimators': 10, 'random_state': 0}
    ]
}
