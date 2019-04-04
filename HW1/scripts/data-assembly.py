# CAPP 30254, Homework 1
# Problem 1: Data Acquisition and Analysis
# Data assembly file
#
# Jonathan Tan
# April 3, 2019

# Setup
import numpy as np
import pandas as pd
from sodapy import Socrata
from secrets import SOCRATA_APP_TOKEN # sensitive information omitted from repo

# Define key API constants
ENDPOINT_URL = "data.cityofchicago.org"
DATASET_ID = "6zsd-86xi"
OUTPUT_FILE = "../data/crime.csv"

# Download data from API
# Expecting 268088 + 266206 = 534294 records
with Socrata(ENDPOINT_URL, SOCRATA_APP_TOKEN) as client:
    results = client.get(DATASET_ID,
                         limit=600000,
                         where="year between '2017' and '2018'")
    crime_df = pd.DataFrame.from_records(results)

# Export to CSV file
crime_df.to_csv(OUTPUT_FILE)

# REFERENCES
#
# 1. Data Source: Chicago Data Portal, Crimes - 2001 to present
# last downloaded April 3, 2019
# https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
#
# 2. API usage code adapted from
# https://dev.socrata.com/foundry/data.cityofchicago.org/6zsd-86xi
