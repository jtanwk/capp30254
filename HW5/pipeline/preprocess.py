# CAPP 30254 Machine Learning for Public Policy
# Homework 5 - Improving the Pipeline, Again
# Pipeline Library - Preprocessing functions

#########
# SETUP #
#########

import math
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#####################
# PRIMARY FUNCTIONS #
#####################


def fill_missing(df, cols, median=False):
    '''
    Takes a df and a list of column names as inputs. Replaces missing values
        for all columns with a function of the remaining data. Uses mean by
        default, but can use median by with 'median=True' parameter.
        Adds a column indicating which rows were imputed.

    TODO: output line saying "Filled in [x] missing values in [var]".

    Input:  df - pandas DataFrame
            cols - list of string column names in df to be filled
    Output: pandas Series with missing numeric data filled in
    '''

    for i in cols:
        new_name = df[i].name + '_imputed'
        df[new_name] = pd.isnull(df[i]).astype('int')
        if median:
            df[i] = df[i].fillna(df[i].median())
        else:
            df[i] = df[i].fillna(df[i].mean())

    return df


def bin_continuous_var(df, var, bin_width=None, num_bins=None):
    '''
    Takes a pandas DataFrame, a string label for a continuous variable, and a
    specified bin width and/or number of bins as inputs, then creates a new
    binned variable based on the provided bin specs and returns a new DataFrame
    with the new variable.

    Inputs: df - pandas DataFrame
            var - string label of a continuous variable to discretize
            bin_width - int size of bin to discretize var by
            num_bins - int number of bins to discretize var by
    Output: new_df - pandas DataFrame with new variable named "[var]_bin"
    '''

    # Only one of bin_width and num_bins can be specified at any one time.
    if bin_width and num_bins:
        raise ValueError('bin_width and num_bins cannot both be specified. Please choose one.')
    elif not bin_width and not num_bins:
        raise ValueError('bin_width and num_bins cannot both be None. Please specify one of them.')

    # Create name for new variable
    new_var = var + '_bin'

    # Create deep copy of df to return; avoid implicitly modifying df in place
    new_df = df.copy(deep=True)

    # Discretizing by bin_width:
    if bin_width:
        new_df[new_var] = pd.cut(new_df[var],
                                 np.arange(start=new_df[var].min(),
                                           stop=new_df[var].max(),
                                           step=bin_width))
        new_df[new_var] = new_df[new_var].astype('str')
    # Discretizing by num_bins:
    else:
        new_df[new_var]= pd.cut(new_df[var], num_bins)

    # Drop original continuous var
    new_df = new_df.drop(labels=[var], axis=1)

    return new_df


def make_dummy_vars(df, var):
    '''
    Wrapper for the pandas get_dummies() method. Takes a pandas DataFrame and
    a string variable label as inputs, and returns a new DataFrame with new
    binary variables for every unique value in var.

    Inputs: df - pandas DataFrame
            var - string label for a categorical variable
    Output: new_df - pandas DataFrame with new variables named "[var]_[value]"
    '''

    new_df = df.copy(deep=True)
    new_df = pd.get_dummies(df, columns=[var], dtype=np.int64)

    return new_df


def split_data(df, label, features=None, test_size=0.3):
    '''
    Takes a pandas DataFrame, a specified label string, and an optional list of
    feature names to retain. Returns two dataframes and two series in order:
    (1) training features, (2) test features, (3) training labels,
    (4) test labels.

    Inputs: df - pandas DataFrame.
            label - string label for the variable of interest.
            features - (optional) list of string feature names to use. If None,
                all features in the dataframe are used.
            drop - (optional) list of string feature names to drop. If None,
                no features are dropped.
            test_size - (optional) float proportion (0 < x < 1) of data to use
                as test set. Default value is 0.3.
    Ouputs: x_train - pandas DataFrame of features for training set.
            x_test - pandas DataFrame of features for test set.
            y_train - pandas Series of labels for training set.
            y_test - pandas Series of labels for test set.

    TODO: Extend to k-fold cross-validation.
    '''

    # Separate label from feature sets
    features_df = df.drop(labels=[label], axis=1)
    labels_df = df[label]

    # If features is None, use all columns except the label
    if features:
        features_df = features_df[features]

    # Split into test and train sets for features and labels
    return train_test_split(features_df, labels_df, test_size=test_size)


def split_data_temporal(df, date_col, split_dicts):
    '''
    Takes a pandas DataFrame and specified label and date column names as
    inputs. Splits the dataframe on a specified timeframe (default test set
    is most recent 1 year), then returns two dataframes and two series in order:
    (1) training features, (2) test features, (3) training labels,
    (4) test labels.

    split_dict should be provided as a list of dictionaries of mm/dd/yyyy, e.g.:
        split_dict = [{
            'train_start': '1/1/2012',
            'train_end': '6/30/2012',
            'test_start': '7/1/2012',
            'test_end': '12/31/2012'
        }]

    Inputs: df - pandas DataFrame.
            date_col - string label for the date column to split on. Must be
                in pandas datetime format.
            split_dicts - list of dictionaries of dates. See above.
    Ouputs: train_dfs - list of pandas DataFrames for training set.
            test_dfs - list of pandas DataFrames for test set.
    '''

    train_dfs = []
    test_dfs = []

    for i in split_dicts:
        train_dfs.append(
            df.loc[ (df[date_col] >= pd.to_datetime(i['train_start'])) &
                    (df[date_col] <= pd.to_datetime(i['train_end']))
        ])
        test_dfs.append(
            df.loc[ (df[date_col] >= pd.to_datetime(i['test_start'])) &
                    (df[date_col] <= pd.to_datetime(i['test_end']))
        ])

    return train_dfs, test_dfs

#
