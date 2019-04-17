# CAPP 30254 Machine Learning for Public Policy
# Homework 2 - Machine Learning Pipeline
#
# Pipeline Configuration file
# Description: This file holds all hard-coded values for the HW2 ML pipeline,
#   including file paths, model parameters, etc. The section headers correspond
#   to the specific portion of the assignment where the particular config
#   variable is used.

################
# 1. READ DATA #
################

# Filepath where credit card data is stored
DATA_PATH = 'data/credit-data.csv'

#######################
# 5. BUILD CLASSIFIER #
#######################

# Proportion of full data to use as a test set
TEST_SIZE = 0.3

# Probability threshold for classifying an observation as positive
CLASS_THRESHOLD = 0.5

# Filepath where outputs of decision tree classifier are stored
OUT_FILE_DOT = 'output/decision_tree.dot'
OUT_FILE_GV = 'output/decision_tree.gv'

# Class names to label on tree
CLASS_NAMES = ['Delinquent', 'Not Delinquent']
