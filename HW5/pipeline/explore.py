# CAPP 30254 Machine Learning for Public Policy
# Homework 5 - Improving the Pipeline, Again
# Pipeline Library - Exploration functions

#########
# SETUP #
#########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#####################
# PRIMARY FUNCTIONS #
#####################


def read_data(csv):
    '''
    Takes a CSV file as input and returns a pandas DataFrame.

    Input: CSV file
    Output: pandas DataFrame of CSV file
    '''

    return pd.read_csv(csv)


def plot_distributions(df, varlist=None):
    '''
    Plots histograms for every variable. Possible to only plot a subset of
    variables by giving list of variables as a parameter.

    Inputs: df - pandas DataFrame.
            varlist - list of strings of varnames to plot. Default is all vars.
    Output: None
    Other: Plots histograms of selected variables in df.

    TODO: extend to non-numeric features
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


def get_correlations(df, plot=False):
    '''
    Prints a correlation table between all variables in the DataFrame.
    If 'plot=True' is specified, plots the correlation matrix instead.

    Input:  df - pandas DataFrame
    Output: None
    Other:  Prints table or plots heatmap of correlations, depending on kwargs.

    TODO: Extend for non-numeric features.
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


def describe_data(df, varlist=None):
    '''
    Wrapper for pandas describe() method; adds easy subsetting by providing
    list of variable names.

    Input: df - pandas DataFrame
    Output: df.describe() - pandas DataFrame of summaries for all numeric vars

    TODO: Extend for non-numeric features.
    '''

    if varlist:
        df = df[varlist]

    return df.describe()

#
