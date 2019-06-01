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

# classifier = (method, param_dict, df_num, trained)

class TrainedClassifier:
    '''
    Object to hold trained classifier object and metadata.
        method - string name of classifier used
        param_dict - dictionary of hyperparameters used to train model
        df_num - integer id for test-train split
        trained - trained classifier object

    The train_classifier() function returns a TrainedClassifier.
    '''

    def __init__(self, method, param_dict, df_num, trained):
        self.method = method
        self.parameters = param_dict
        self.df_num = df_num
        self.classifier = trained


def train_classifier(df, label, method, df_num, param_dict=None):
    '''
    Takes a pandas DataFrame, name of a label feature, the name of a classifier
    to fit, and an optional dictionary of classifier hyperparameters as inputs.

    Returns a TrainedClassifier object with attributes: method, parameters,
        df_num, classifier.

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
    Output: TrainedClassifier object
    '''
    print(str(datetime.datetime.now()) + ' Training ' + method + \
        ' with params ' + str(param_dict) + ' on training set '  + str(df_num))

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
        print(method + 'is not a supported classifier. Please use one of:')
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

    # Store results in TrainedClassifier object
    result = TrainedClassifier(method, param_dict, df_num, trained)

    return result


def evaluate_classifier(df, label, classifier, top_k):
    '''
    Takes a test dataframe with features and labels, a tuple returned from
    train_classifier() method, and an optional threshold k for precision-recall.
    Calculates several evaluation metrics (accuracy, precision, recall, F1,
    etc.) and returns a dictionary of those metrics.

    Inputs: df - pandas DataFrame of features and label for test set
            classifier - a TrainedClassifier object from train_classifier()
            label - string name of feature in df to predict on
            top_k - list of top percentages of ranked obs to classify as 1
    Output: dataframe of evaluation metrics for the given classifier.
    '''

    # classifier = (method, param_dict, df_num, trained)
    print(
        str(datetime.datetime.now()) + ' Evaluating ' + classifier.method \
        + ' with ' + str(classifier.parameters) + ' on test set ' \
        + str(classifier.df_num)
    )

    # Get predicted scores; need to manually get scores from LinearSVC
    x_test = df.drop(labels=[label], axis=1)
    if isinstance(classifier.classifier, LinearSVC):
        y_scores = classifier.classifier.decision_function(x_test)
    else:
        y_scores = classifier.classifier.predict_proba(x_test)[:, 1]

    # For a given percentage, label the top n observations as 1
    pred_df = pd.DataFrame({'label': df[label], 'score': y_scores}) \
        .sort_values(by=['score'], ascending=False).reset_index(drop=True)

    result_df = pd.DataFrame()
    y_test = pred_df['label']
    for k in top_k:
        top_n = math.ceil(k * len(df))
        y_pred = np.where(df.index < top_n, 1, 0)

        # Calculate accuracy, precision, recall, f1, auc-roc for a given k
        results_dict = {
            'classifier': classifier.method,
            'params': classifier.parameters,
            'k': str(k * 100) + "%",
            'test-train-id': classifier.df_num,
            'accuracy': accuracy_score(y_true=y_test, y_pred=y_pred),
            'precision': precision_score(y_true=y_test, y_pred=y_pred),
            'recall': recall_score(y_true=y_test, y_pred=y_pred),
            'f1': f1_score(y_true=y_test, y_pred=y_pred),
            'auc-roc': roc_auc_score(y_true=y_test, y_score=y_scores)
        }
        result_df = result_df.append(results_dict, ignore_index=True)

    return result_df


#
