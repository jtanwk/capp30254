# Homework 1 - Diagnostic

## What file do I look at?
Please look at the [IPython notebook](https://github.com/jtanwk/capp30254/blob/master/HW1/Homework%201.ipynb) for all code and writeup.

## Notes for this submission
- Given the large volume of crime data (over 500,000 rows), the data was downloaded via the Socrata Open Data API and stored in `/data/crime.csv`. The script to replicate this can be found in `/scripts/data-assembly.py`.
- I used a private key to download data from the Census API. Since this information is sensitive, I stored it in an environment variable for this Jupyter notebook. If you have your own Census API key, you can set it by running `%env CENSUS_KEY [YOUR-KEY-HERE]` before the rest of the notebook.
