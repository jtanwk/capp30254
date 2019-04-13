# CAPP 30254 Machine Learning for Public Policy
# Homework 2 - Machine Learning Pipeline
# Pipeline Library file

#########
# SETUP #
#########

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


################
# 1. READ DATA #
################

def read_data(csv):
    '''
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
# 'Outlier' here is defined (if numeric) as having value of 1.5 * IQR

# TODO


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

    # Create copy of df to return; avoid implicitly modifying in place
    new_df = df.copy(deep=True)

    # Discretizing by bin_width:
    if bin_width:
        new_df[new_var] = pd.cut(new_df[var],
                                 np.arange(start=new_df[var].min(),
                                           stop=new_df[var].max(),
                                           step=bin_width))
    else: # Discretizing by num_bins:
        new_df[new_var]= pd.cut(new_df[var], num_bins)

    return new_df


# 4B. Make dummy variables from categorical variable:
def make_dummy_vars(df, var):
    '''
    Wrapper for the pandas get_dummies() method. Takes a pandas DataFrame and
    a string variable label as inputs, and returns a new DataFrame with new
    binary variables for every unique value in var.

    Inputs: df - pandas DataFrame
            var - string label for a categorical value
    Output: new_df - pandas DataFrame with new variables named "[var]_[value]"
    '''

    # Create copy of df to return; avoid implicitly modifying in place.
    new_df = df.copy(deep=True)

    # Get dummy variables
    new_df = pd.get_dummies(df, columns=[var])

    return new_df




#
