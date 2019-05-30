# CAPP 30254 Machine Learning for Public Policy
# Homework 5 - Improving the Pipeline, Again
# Main Pipeline Executable

#########
# SETUP #
#########

import math
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, scale

import config as cf
from pipeline.explore import read_data
from pipeline.preprocess import fill_missing, bin_continuous_var, \
                                make_dummy_vars, split_data_temporal
from pipeline.testtrain import train_classifier, validate_classifier


#############
# MAIN BODY #
#############

def main():

    # read data
    print('reading data')
    df = pd.read_csv(cf.DATA_PATH,
                     parse_dates=['date_posted', 'datefullyfunded'])


    # select features
    selected_features = ['school_state', 'school_metro', 'school_charter',
       'school_magnet', 'teacher_prefix', 'primary_focus_area',
       'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level',
       'total_price_including_optional_support', 'students_reached',
       'eligible_double_your_impact_match', 'date_posted', 'datefullyfunded']
    df = df[selected_features]


    # split into test-train
    print('splitting data')
    train_dfs, test_dfs = split_data_temporal(df=df,
                                              date_col=cf.DATE_COL,
                                              split_dicts=cf.TEMPORAL_SPLITS)


    # define label
    print('defining label')
    for df_list in (train_dfs, test_dfs):
        for df in df_list:

            # Label = 1 if datefullyfunded is more than 60 days after date_posted
            df['not_funded_60_days'] = np.where(
                df['datefullyfunded'] - df['date_posted'] > \
                    pd.to_timedelta(60, unit='days'), 1, 0)
            # Leave a lag period of 60 days at the end of each dataset
            df = df.loc[df['date_posted'].max() - df['date_posted'] > \
                pd.to_timedelta(60, unit='days')]

    # Drop unnecessary columns
    for df_list in (train_dfs, test_dfs):
        for i in range(len(df_list)):

            df_list[i] = df_list[i].drop(labels=['date_posted', 'datefullyfunded'], axis=1)


    # clean missing data
    print('cleaning data')
    for df_list in (train_dfs, test_dfs):
        for df in df_list:
            df = fill_missing(df, ['students_reached'], median=True)


    # standardize numeric data
    numeric_features = ['total_price_including_optional_support', 'students_reached']
    for df_list in (train_dfs, test_dfs):
        for df in df_list:

            for i in numeric_features:
                df[i] = scale(df[i])


    # make string binary into true binary
    binary_features = ['school_charter', 'school_magnet',
                       'eligible_double_your_impact_match']
    for df_list in (train_dfs, test_dfs):
        for df in df_list:

            for i in binary_features:
                df[i] = np.where(df[i] == 't', 1, 0)


    # transform categorical features into dummies
    categorical_features = ['school_state', 'school_metro', 'teacher_prefix',
                        'resource_type', 'primary_focus_area',
                        'secondary_focus_area', 'poverty_level', 'grade_level']
    for df_list in (train_dfs, test_dfs):
        for i in range(len(df_list)):

            for j in categorical_features:
                df_list[i] = make_dummy_vars(df_list[i], j)


    # make sure features match between test and train sets
    test_dfs[2] = test_dfs[2].drop(labels=['teacher_prefix_Dr.'], axis=1)


    # train classifiers
    classifiers = cf.CLASSIFIERS # list of string names of classifiers
    parameters = cf.GRID_LARGE # dictionary of lists of parameters
    num_training_sets = len(cf.TEMPORAL_SPLITS) # use to index into train_dfs
    label = cf.LABEL
    trained_classifiers = []

    for i in classifiers:
        for j in parameters[i]:
            for k in range(num_training_sets):

                trained = train_classifier(df=train_dfs[k],
                                           label=label,
                                           method=i,
                                           df_num=k,
                                           param_dict=j)
                trained_classifiers.append(trained)

    # evaluate classifiers
    thresholds = cf.THRESHOLDS
    results_df = pd.DataFrame()

    for i in trained_classifiers: # (method, param_dict, df_num, trained)
        for j in thresholds:

            results_dict = validate_classifier(df=test_dfs[i[2]],
                                               label=cf.LABEL,
                                               classifier=i,
                                               top_k=j)
            results_df = results_df.append(results_dict, ignore_index=True)

    # save results to csv
    COL_ORDER = ['classifier', 'params', 'k', 'test-train-id', 'accuracy', 'precision', 'recall', 'f1', 'auc-roc']
    results_df[COL_ORDER].to_excel("output/results.xlsx")

    print(f"COMPLETED AT {datetime.datetime.now()}")


if __name__ == '__main__':
    main()


#
