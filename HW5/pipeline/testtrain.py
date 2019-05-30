# CAPP 30254 Machine Learning for Public Policy
# Homework 5 - Improving the Pipeline, Again
# Pipeline Library - Training and Evaluation functions

#########
# SETUP #
#########

import math
import datetime
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
                             BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, precision_recall_curve, roc_auc_score


#####################
# PRIMARY FUNCTIONS #
#####################


def train_classifier(df, label, method, df_num, param_dict=None):
    '''
    Takes a pandas DataFrame, name of a label feature, the name of a classifier
    to fit, and an optional dictionary of classifier hyperparameters as inputs.

    Returns a trained classifier object and related information.

    Inputs: df - pandas DataFrame of training data containing a label feature
            label - string name of the label feature to train on
            method - string name of classifiers to use. Must be one of:
                         1. LogisticRegression
                         2. KNeighborsClassifier
                         3. DecisionTreeClassifier
                         4. LinearSVC
                         5. RandomForestClassifier
                         6. AdaBoostClassifier
                         7. BaggingClassifier
            param_dict - (optional) dictionary of parameters to initialize each
                classifier with. If None, uses small test params loop.
    Output: (method, param_dict, trained) - a 4-tuple of (1) classifier name,
        (2) hyperparameters used, (3) the test/train id, and (4) the trained
        classifier object.
    '''
    print(str(datetime.datetime.now() + ' Training ' + method +
        'with params' + param_dict + 'on training set'  + df_num))

    # Supported classifiers
    method_dict = {
        'LogisticRegression': LogisticRegression,
        'KNeighborsClassifier': KNeighborsClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'LinearSVC': LinearSVC,
        'RandomForestClassifier': RandomForestClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'BaggingClassifier': BaggingClassifier
    }

    # Test parameter dictionary - simple hyperparameters
    param_defaults = {
        'LogisticRegression': {'penalty': 'l2', 'C': 1, 'random_state': 0},
        'KNeighborsClassifier': {'n_neighbors': 1, 'weights': 'uniform', 'algorithm': 'auto'},
        'DecisionTreeClassifier': {'max_depth': 1, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'random_state': 0},
        'LinearSVC': {'penalty': 'l2', 'C': 1, 'random_state': 0},
        'RandomForestClassifier': {'n_estimators': 10, 'max_depth': 1, 'max_features': 10, 'min_samples_leaf': 10, 'random_state': 0},
        'AdaBoostClassifier': {'n_estimators': 10, 'algorithm': 'SAMME.R', 'random_state': 0},
        'BaggingClassifier': {'n_estimators': 10, 'random_state': 0}
    }

    # Raise error if provided method is not supported
    if method not in method_dict.keys():
        print(method + 'is not a supported classifier. Pleaes use one of:')
        print(method_dict.keys())
        return None

    # Initilize classifier with supplied dictionary, else defaults
    if param_dict:
        classifier = method_dict[method](**param_dict)
    else:
        classifier = method_dict[method](**param_defaults[method])

    # Split data into features/labels and train classifier
    x_train = df.drop(labels=[label], axis=1)
    y_train = df[label]
    trained = classifier.fit(x_train, y_train)

    return (method, param_dict, df_num, trained)


def validate_classifier(df, label, classifier, top_k):
    '''
    Takes a test dataframe with features and labels, a tuple returned from
    train_classifier() method, and an optional threshold k for precision-recall.
    Calculates several evaluation metrics (accuracy, precision, recall, F1,
    etc.) and returns a dictionary of those metrics.

    Inputs: df - pandas DataFrame of features and label for test set
            classifier - 4-tuple of (method, params, df_id, classifier).
                Classifier object must be one of:
                 1. LogisticRegression
                 2. KNeighborsClassifier
                 3. DecisionTreeClassifier
                 4. LinearSVC
                 5. RandomForestClassifier
                 6. AdaBoostClassifier
                 7. BaggingClassifier
            label - string name of feature in df to predict on
            top_k - top percentage of ranked observations to classify as positive
    Output: dictionary of evaluation metrics for the given classifier.
    '''

    # classifier = (method, param_dict, df_num, trained)
    print(str(datetime.datetime.now() + 'Evaluating ' + classifier[0] + ' with ' \
        + classifier[1] + ' on top ' +  str(top_k * 100) + '%'})

    # Initialize dictionary to store results
    results_dict = {
        'classifier': classifier[0],
        'params': classifier[1],
        'k': str(top_k * 100) + "%",
        'test-train-id': classifier[2]
    }

    # Get predicted scores; need to manually get scores from LinearSVC
    x_test = df.drop(labels=[label], axis=1)
    if isinstance(classifier[3], LinearSVC):
        y_scores = classifier[3].decision_function(x_test)
    else:
        y_scores = classifier[3].predict_proba(x_test)[:, 1]

    # For a given percentage, label the top n observations as 1
    pred_df = pd.DataFrame({'label': df[label], 'score': y_scores}) \
        .sort_values(by=['score'], ascending=False).reset_index(drop=True)
    top_n = math.ceil(top_k * len(df))
    y_pred = np.where(df.index < top_n, 1, 0)
    y_test = pred_df['label']

    # Calculate accuracy, precision, recall, f1, auc-roc for a given k
    results_dict['accuracy'] = accuracy_score(y_true=y_test, y_pred=y_pred)
    results_dict['precision'] = precision_score(y_true=y_test, y_pred=y_pred)
    results_dict['recall'] = recall_score(y_true=y_test, y_pred=y_pred)
    results_dict['f1'] = f1_score(y_true=y_test, y_pred=y_pred)
    results_dict['auc-roc'] = roc_auc_score(y_true=y_test, y_score=y_scores)

    return results_dict


#
