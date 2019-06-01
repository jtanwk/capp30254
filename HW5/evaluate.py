# CAPP 30254 Machine Learning for Public Policy
# Homework 5 - Improving the Pipeline, Again
# Evaluation only (to prevent retraining)

import datetime
import pickle
import argparse

import numpy as np
import pandas as pd

import config as cf
from pipeline.testtrain import evaluate_classifier


def main():

    # announce test or full mode
    if args.test:
        print(str(datetime.datetime.now()) + " Running test grid ")
        parameters = cf.GRID_TEST # simple test grid with 3 classifiers
    else:
        print(str(datetime.datetime.now()) + " Running full grid ")
        parameters = cf.GRID_MAIN # full grid

    # load stored test/train data
    with open(cf.TEST_TRAIN_PATH, 'rb') as f:
        train_dfs, test_dfs = pickle.load(f)

    # load stored trained classifiers
    with open(cf.CLASSIFIER_PATH, 'rb') as f:
        trained_classifiers = pickle.load(f)

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
