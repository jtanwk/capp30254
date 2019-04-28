# CAPP 30254 Machine Learning for Public Policy
# Homework 3 - Improving the Pipeline
#
# Pipeline Library file
# Description: tbd

#########
# SETUP #
#########

# Import useful libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
                             AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, precision_recall_curve

################
# 1. READ DATA #
################

def read_data(csv):
    '''
    Takes a CSV file as input and returns a pandas DataFrame.

    Input: CSV file
    Output: pandas DataFrame of CSV file
    '''

    return pd.read_csv(csv)


###################
# 2. EXPLORE DATA #
###################

# 1. Generate distributions of variables
def plot_distributions(df, varlist=None):
    '''
    Plots histograms for every variable. Possible to only plot a subset of
    variables by giving list of variables as a parameter.

    Inputs: df - pandas DataFrame.
            varlist - list of strings of varnames to plot. Default is all vars.
    Output: None
    Other: Plots histograms of selected variables in df.
    '''

    # Filter dataframe down by selected variables, if any.
    if varlist:
        df = df[varlist]

    categories = df.columns.tolist()

    # Set up dimensions for empty figure; fixed width, variable height.
    num_plots = len(categories)
    num_rows, NUM_COLS = (num_plots // 3) + 1, 3
    FIG_WIDTH, fig_height = 16, 4 * num_rows

    # Create empty figure.
    fig = plt.figure(figsize=(FIG_WIDTH, fig_height))
    axes = [plt.subplot(num_rows, NUM_COLS, i) for i in range(1, num_plots + 1)]
    plt.tight_layout(pad=0, w_pad=1, h_pad=3)

    # Fill figure with histograms.
    for i in range(num_plots):
        ax = axes[i]
        df[categories[i]].hist(
            ax=ax,
            grid=False)

        # Set title.
        ax.set_title(categories[i])

        # Label bars.
        for p in ax.patches:
            ax.annotate(str(p.get_height()),
                        (p.get_x(), p.get_height()))

    # Display figure.
    plt.show()


# 2. Find correlations between variables
# Adapted from https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
def get_correlations(df, plot=False):
    '''
    Prints a correlation table between all variables in the DataFrame.
    If 'plot=True' is specified, plots the correlation matrix instead.

    Input:  df - pandas DataFrame
    Output: None
    Other:  Prints table or plots heatmap of correlations, depending on kwargs.
    '''

    if plot:
        # Setup plot
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(15, 15))

        # Populate plot with matrix of correlations
        ax.matshow(corr)

        # Apply variable labels, rotate where necessary
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.xticks(rotation=30,
                   rotation_mode='anchor',
                   ha='left')

        # Display plot
        plt.show()
    else:
        print(df.corr())


# 3. Find outliers in numeric variables
def get_outliers(df, var):
    '''
    Takes a pandas DataFrame and string variable as inputs, and returns
    a DataFrame only containing rows with outlier values for the specified
    variable. Only applies to numeric vars.

    Inputs: df - pandas DataFrame
            var - string variable name to find outliers in
    Output: new_df - pandas DataFrame with only outlier rows
    '''

    # Create deep copy of df to return; avoid implicitly modifying df in place
    new_df = df[[var]]
    new_df = df.copy(deep=True)

    # Find bounds for outliers
    q1, q3 = np.nanpercentile(new_df, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # identify outliers
    is_outlier = lambda x: x < lower_bound or x > upper_bound

    return df.loc[df[var].apply(is_outlier)]



# 4. Summarize numeric data
def describe_data(df, varlist=None):
    '''
    Wrapper for pandas describe() method; adds easy subsetting by providing
    list of variable names.

    Input: df - pandas DataFrame
    Output: df.describe() - pandas DataFrame of summaries for all numeric vars
    '''

    if varlist:
        df = df[varlist]

    return df.describe()


#######################
# 3. PRE-PROCESS DATA #
#######################

def fill_missing(df, median=False):
    '''
    Takes a df and replaces missing values for all numeric variables with a
    function of the remaining data. Function is mean by default, but can use
    median by giving 'median=True' parameter.

    TODO: output line saying "Filled in [x] missing values in [var]".

    Input: df - pandas DataFrame
    Output: filled_df - pandas DataFrame with missing numeric data filled in
    '''

    if median:
        return df.fillna(df.median())
    else:
        return df.fillna(df.mean())


###################################
# 4. GENERATE FEATURES/PREDICTORS #
###################################

# 4A. Discretize continuous variable.
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

# 4B. Make dummy variables from categorical variable:
def make_dummy_vars(df, var):
    '''
    Wrapper for the pandas get_dummies() method. Takes a pandas DataFrame and
    a string variable label as inputs, and returns a new DataFrame with new
    binary variables for every unique value in var.

    Inputs: df - pandas DataFrame
            var - string label for a categorical variable
    Output: new_df - pandas DataFrame with new variables named "[var]_[value]"
    '''

    # Create copy of df to return; avoid implicitly modifying in place.
    new_df = df.copy(deep=True)

    # Get dummy variables
    new_df = pd.get_dummies(df, columns=[var])

    return new_df


# 4C. Encode ordinal variables into numeric format
def encode_ordinal_var(df, var, value_dict):
    '''
    Encodes a categorical, ordinal variable into an integer representation
    that preserves the underlying meaning of the variable.

    Inputs: df - pandas DataFrame
            var - string label for an ordinal variable
            value_dict - dictionary of values to assign to variable
    Output: new_df - pandas DataFrame with transformed ordinal variable
    '''

    pass


########################
# 5. Build Classifiers #
########################

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
    '''

    # Separate label from feature sets
    features_df = df.drop(labels=[label], axis=1)
    labels_df = df[label]

    # If features is None, use all columns except the label
    if features:
        features_df = features_df[features]

    # Split into test and train sets for features and labels
    return train_test_split(features_df, labels_df, test_size=test_size)


def split_data_temporal(df, label, date_col, test_dur=1, test_units='Y'):
    '''
    Takes a pandas DataFrame and specified label and date column names as
    inputs. Splits the dataframe on a specified timeframe (default test set
    is most recent 1 year), then returns two dataframes and two series in order:
    (1) training features, (2) test features, (3) training labels,
    (4) test labels.

    Inputs: df - pandas DataFrame.
            label - string label for the variable of interest.
            date_col - string label for the date column to split on. Must be
                in pandas datetime format.
            test_dur - integer value for length of test set. Default is 1.
            test_units - units for test_dur. Default is 'Y' (years)
    Ouputs: x_train - pandas DataFrame of features for training set.
            x_test - pandas DataFrame of features for test set.
            y_train - pandas Series of labels for training set.
            y_test - pandas Series of labels for test set.
    '''

    # Define threshold date based on specified test set length
    threshold = df[date_col].max() - pd.to_timedelta(test_dur, test_units)

    # Training data is everything before threshold, test data is after
    x_train = df.loc[df[date_col] <= threshold]
    x_test = df.loc[df[date_col] > threshold]

    # Separate out labels for train and test sets
    y_train = x_train[label]
    x_train = x_train.drop(labels=[label], axis=1)
    y_test = x_test[label]
    x_test = x_test.drop(labels=[label], axis=1)

    return x_train, x_test, y_train, y_test


def train_classifier(x_train, y_train, method=None, param_dict=None):
    '''
    Takes 2 pandas DataFrames (features and labels of training data) and an
    optional list of classifiers to fit. Returns a dictionary of trained
    classifiers.

    Inputs: x_train - pandas DataFrame of features for training set
            x_test - pandas DataFrame of features for test set
            method - (optional) list of string names of classifiers to use.
                        If None, will use all of the following:
                        lr      = LogisticRegression
                        knn     = KNeighborsClassifier
                        dt      = DecisionTreeClassifier
                        svm     = LinearSVC
                        rf      = RandomForestClassifier
                        boost   = AdaBoostClassifier
                        bag     = BaggingClassifier
            model_params - (optional) nested dictionary of parameters to
                        initialize each classifier with. If None, uses sklearn
                        defaults.
    Output: classifier_dict - dictionary of trained classifier objects.
    '''

    # If method is None, use this preset list of classifiers
    method_dict = {
        'lr': LogisticRegression,
        'knn': KNeighborsClassifier,
        'dt': DecisionTreeClassifier,
        'svm': LinearSVC,
        'rf': RandomForestClassifier,
        'boost': AdaBoostClassifier,
        'bag': BaggingClassifier
    }

    if not method:
        method = method_dict.keys()

    # Fit all selected classifiers and append to dictionary to return
    classifier_dict = {}

    for i in method:
        print(f'Training {i} with params {param_dict[i]}')

        if not param_dict:
            classifier = method_dict[i]()
        else:
            params = param_dict[i]
            classifier = method_dict[i](**params)

        classifier_dict[i] = classifier.fit(x_train, y_train)

    return classifier_dict


###########################
# 6. Validate Classifiers #
###########################


def validate_classifier(x_test, y_test, classifier_dict, threshold=0.5):
    '''
    Takes 2 dataframes (features and labels for test data) and a dictionary of
    pre-trained classifiers. When called, prints a series of evaluation
    metrics for each classifier: accuracy, precision, recall, F1, AUC-PR.

    Inputs: x_test - pandas DataFrame of features for test set
            y_test - pandas Series of labels for test set
            classifier_dict - dictionary of trained classifiers.
                Possible key/value pairs are as follows:
                    lr      = LogisticRegression
                    knn     = KNeighborsClassifier
                    dt      = DecisionTreeClassifier
                    svm     = LinearSVC
                    rf      = RandomForestClassifier
                    boost   = AdaBoost
                    bag     = BaggingClassifier
            threshold - (optional) float threshold to use in predicting labels
                based on calculated scores from classifiers.

    Output: Returns none. Prints a dataframe summarizing evaluation metrics.
    '''

    # CLASSIFIER, PARAM 1, PARAM 2, TEST SPLIT, FEATURES USED, ACCURACY
    # PRECISION AT 1%, AUC

    # Define quick function to compare score against threshold
    calc_threshold = lambda x, y: 0 if x < y else 1
    accuracy, precision, recall, f1 = [], [], [], []

    # Iterate over classifiers, predict values, and store results
    for key in classifier_dict.keys():
        print(f'Validating {key}')

        # LinearSVC classifier does not output a score, only labels
        if key == 'svm':
            y_pred = classifier_dict[key].predict(x_test)
        else:
            y_scores = classifier_dict[key].predict_proba(x_test)[:, 1]
            y_pred = pd.Series([calc_threshold(x, threshold) for x in y_scores])

        # log results
        accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))
        precision.append(precision_score(y_true=y_test, y_pred=y_pred))
        recall.append(recall_score(y_true=y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=y_test, y_pred=y_pred))

    return pd.DataFrame(list(
        zip(classifier_dict.keys(), accuracy, precision, recall, f1)),
        columns=['classifier', 'accuracy', 'precision', 'recall', 'f1']
    )

    # For test-train splits (CV or temporal),
    # For subsets of feature sets (demographic only, behavior only, temporal only)
    # For classifiers
    # For parameters (cross-product of parameters)

    # Fit
    # Predict
    # Evaluate

















#
