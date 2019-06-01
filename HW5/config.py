# CAPP 30254 Machine Learning for Public Policy
# Homework 5 - Improving the Pipeline
# Pipeline Configuration file


######################
# 1. READ/WRITE DATA #
######################

# Filepath where input data is stored
DATA_PATH = 'data/projects_2012_2013.csv'

# Filepath where trained classifiers are stored
CLASSIFIER_PATH = 'output/trained_classifiers.pkl'

# Filepath where cleaned test/train data are stored
TEST_TRAIN_PATH = 'output/test_train_clean.pkl'

# Identifying column of interest
LABEL = 'not_funded_60_days'
DATE_COL = 'date_posted'


########################
# 3. TEST/TRAIN SPLITS #
########################

# Dates for temporal test/train splits
TEMPORAL_SPLITS = [
    {
        'train_start': '1/1/2012',
        'train_end': '6/30/2012',
        'test_start': '7/1/2012',
        'test_end': '12/31/2012'
    },
    {
        'train_start': '1/1/2012',
        'train_end': '12/31/2012',
        'test_start': '1/1/2013',
        'test_end': '6/30/2013'
    },
    {
        'train_start': '1/1/2012',
        'train_end': '6/30/2013',
        'test_start': '7/1/2013',
        'test_end': '12/31/2013'
    },
]

#######################
# 5. BUILD CLASSIFIER #
#######################

# Large grid - most exhaustive option
GRID_MAIN = {
    'classifiers': ['LogisticRegression', 'KNeighborsClassifier',
                   'DecisionTreeClassifier', 'LinearSVC',
                   'RandomForestClassifier', 'AdaBoostClassifier',
                   'BaggingClassifier'],
    'thresholds': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1],
    'LogisticRegression': [
        {'penalty': x, 'C': y, 'solver': 'liblinear', 'random_state': 0} \
        for x in ('l1', 'l2') \
        for y in (0.01, 0.1, 1, 10, 100) \
    ],
    'KNeighborsClassifier': [
        {'n_neighbors': x, 'weights': y, 'algorithm': z, 'n_jobs': -1} \
        for x in (5, 10, 50) \
        for y in ('uniform', 'distance') \
        for z in ('auto', 'ball_tree', 'kd_tree')
    ],
    'DecisionTreeClassifier': [
        {'max_depth': x, 'max_features': y, 'min_samples_leaf': z,
        'random_state': 0} \
        for x in (5, 10, 50) \
        for y in ('sqrt', 'log2', None) \
        for z in (5, 10)
    ],
    'LinearSVC': [
        {'penalty': 'l2', 'C': x, 'random_state': 0} \
        for x in (0.01, 0.1, 1, 10, 100)
    ],
    'RandomForestClassifier': [
        {'n_estimators': x, 'max_depth': y, 'max_features': z,
        'random_state': 0, 'n_jobs': -1} \
        for x in (10, 100, 1000) \
        for y in (5, 10, 50) \
        for z in ('sqrt', 'log2')
    ],
    'AdaBoostClassifier': [
        {'n_estimators': x, 'algorithm': y, 'random_state': 0} \
        for x in (10, 100, 1000) \
        for y in ('SAMME', 'SAMME.R')
    ],
    'BaggingClassifier': [
        {'n_estimators': x, 'random_state': 0, 'n_jobs': -1} \
        for x in (10, 100, 1000)
    ]
}

# Test grid to make sure everything works - 1 model per classifier
GRID_TEST = {
    'classifiers': ['LogisticRegression', 'KNeighborsClassifier',
                   'DecisionTreeClassifier'],
    'thresholds': [0.5],
    'LogisticRegression': [
        {'penalty': 'l2', 'C': 1, 'solver': 'liblinear', 'random_state': 0}
    ],
    'KNeighborsClassifier': [
        {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto',
         'n_jobs': -1}
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
         'random_state': 0, 'n_jobs': -1}
    ],
    'AdaBoostClassifier': [
        {'n_estimators': 10, 'algorithm': 'SAMME.R', 'random_state': 0}
    ],
    'BaggingClassifier': [
        {'n_estimators': 10, 'random_state': 0, 'n_jobs': -1}
    ]
}
