# CAPP 30254 Machine Learning for Public Policy
# Homework 5 - Improving the Pipeline, Again
# Main Pipeline Executable

#########
# SETUP #
#########
import math
import datetime
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, scale

import config as cf
from pipeline.explore import read_data
from pipeline.preprocess import fill_missing, bin_continuous_var, \
                                balance_features, make_dummy_vars, \
                                split_data_temporal
from pipeline.testtrain import train_classifier, evaluate_classifier


########################
# MAIN PIPELINE STAGES #
########################

def select_features(df):

    # Define selected features
    selected= ['school_state', 'school_metro', 'school_charter', 'school_magnet',
        'teacher_prefix', 'primary_focus_area', 'secondary_focus_area',
        'resource_type', 'poverty_level', 'grade_level',
        'total_price_including_optional_support', 'students_reached',
        'eligible_double_your_impact_match', 'date_posted', 'datefullyfunded']

    return df[selected]


def define_label(df):

    # Label = 1 if datefullyfunded is more than 60 days after date_posted
    df['not_funded_60_days'] = np.where(df['datefullyfunded'] - df['date_posted'] > \
            pd.to_timedelta(60, unit='days'), 1, 0)
    # Leave a lag period of 60 days at the end of each dataset
    df = df.loc[df['date_posted'].max() - df['date_posted'] > \
            pd.to_timedelta(60, unit='days')] \
        .drop(labels=['date_posted', 'datefullyfunded'], axis=1)

    return df


def clean_data(df):

    # clean missing data
    df = fill_missing(df, ['students_reached'], median=True)

    # standardize numeric features to mean 0 sd 1
    numeric_features = ['total_price_including_optional_support', 'students_reached']
    for i in numeric_features:
        df[i] = scale(df[i])

    # make string binary (t/f) into true binary (1/0)
    binary_features = ['school_charter', 'school_magnet', 'eligible_double_your_impact_match']
    for i in binary_features:
        df[i] = np.where(df[i] == 't', 1, 0)

    # transform categorical features into dummies
    categorical_features = ['school_state', 'school_metro', 'teacher_prefix',
        'resource_type', 'primary_focus_area', 'secondary_focus_area',
        'poverty_level', 'grade_level']
    for i in categorical_features:
        df = make_dummy_vars(df, i)

    return df


def save_to_file(obj, path):

    # Save passed obj as a pickle to given filepath
    with open(path, 'wb') as f:
        pickle.dump(obj=obj,
                    file=f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(str(datetime.datetime.now()) + " Saving object to " + path)
    return None


def main():

    # announce test or full mode
    if args.test:
        print(str(datetime.datetime.now()) + " Running test grid ")
        parameters = cf.GRID_TEST # simple test grid with 3 classifiers
    else:
        print(str(datetime.datetime.now()) + " Running full grid ")
        parameters = cf.GRID_MAIN # full grid

    # read data
    print(str(datetime.datetime.now()) + ' Reading data ')
    df = pd.read_csv(cf.DATA_PATH,
                     parse_dates=['date_posted', 'datefullyfunded'])

    # select features
    df = select_features(df)

    # split into test-train
    print(str(datetime.datetime.now()) + ' Splitting data ')
    train_dfs, test_dfs = split_data_temporal(df=df,
                                              date_col=cf.DATE_COL,
                                              split_dicts=cf.TEMPORAL_SPLITS)

    # preprocessing loop for all datasets
    for df_list in (train_dfs, test_dfs):
        for i in range(len(df_list)):

            # define label
            print(str(datetime.datetime.now()) + ' Defining label for test-train set ' + str(i))
            df_list[i] = define_label(df_list[i])

            # clean data
            print(str(datetime.datetime.now()) + ' Cleaning data for test-train set ' + str(i))
            df_list[i] = clean_data(df_list[i])

    # make sure features match between test and train sets
    for i in range(len(train_dfs)):
        print(str(datetime.datetime.now()) + ' Balancing features for test-train set' + str(i))
        train_dfs[i], test_dfs[i] = balance_features(train_dfs[i], test_dfs[i])

    # save to file
    test_train_data = [train_dfs, test_dfs]
    save_to_file(test_train_data, cf.TEST_TRAIN_PATH)

    # train classifiers
    trained_classifiers = []
    for i in parameters['classifiers']:
        for j in parameters[i]:
            for k in range(len(cf.TEMPORAL_SPLITS)):
                trained = train_classifier(df=train_dfs[k],
                                           label=cf.LABEL,
                                           method=i,
                                           df_num=k,
                                           param_dict=j)
                trained_classifiers.append(trained)

    # save to file
    save_to_file(trained_classifiers, cf.CLASSIFIER_PATH)

    # evaluate classifiers
    results_df = pd.DataFrame()
    for i in trained_classifiers: # i is a TrainedClassifier object
        eval_df = evaluate_classifier(df=test_dfs[i.df_num],
                                      label=cf.LABEL,
                                      classifier=i,
                                      top_k=parameters['thresholds'])
        results_df = results_df.append(eval_df, ignore_index=True)

    # save results to csv
    COL_ORDER = ['classifier', 'params', 'k', 'test-train-id', 'accuracy', 'precision', 'recall', 'f1', 'auc-roc']
    if args.test:
        results_df[COL_ORDER].to_excel("output/results.xlsx")
    else:
        results_df[COL_ORDER].to_excel("output/results_test.xlsx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Use test grid instead of main')
    args = parser.parse_args()

    main()

#
