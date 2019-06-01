# Homework 5 - Improving the Pipeline, Again

## What files do I look at?
- `/pipeline` is the library that contains the modules for my pipeline functions.
    - `explore.py` contains the functions I wrote for reading data, exploration, and visualization.
    - `preprocess.py` contains the functions I wrote for imputation, feature generation, discretization, and other preprocessing steps.
    - `testtrain.py` contains the functions I wrote for model training and validation.
- `config.py` is the python file containing hard-coded parameters used throughout the pipeline, including small and large grids.
- `pipeline.py` is a script version of the full machine learning pipeline that, when run, goes through every stage of the pipeline, and finally exports the grid search results to `/output/results.xlsx`. Can be run with the `--test` argument (e.g. `python3 pipeline.py --test`) to train on a test grid of classifiers and thresholds.
- `evaluate.py` takes saved test/training data and classified models and runs only the evaluation steps in `pipeline.py`. This is used if `pipeline.py` completed the training phase but ran into errors in the evaluation phase to prevent time-consuming re-training.
- `Homework5_Tan.ipynb` demonstrates each step in `pipeline.py` in notebook form, including my steps taken to explore the data and verify that preprocessing steps occurred successfully. It documents critical choices around feature selection, training, etc. for predicting which DonorsChoose projects will not be funded within 60 days of posting. Lastly, it includes a short report of recommendations based on my results.

## Other notes for this submission
- The `/data` directory stores data used in this assignment.
- The `/output` directory stores graph outputs from various models, as well as an excel file of the final table of classifiers and evaluation metrics.
- All `.sbatch` files are shell scripts for submitting jobs on UChicago's RCC Midway compute cluster.

## Running the code

Without using a virtual environment:

    - Directly install all requirements by running `pip install --user -r requirements.txt`.
    - Run `python3 pipeline.py`.

With a virtual environment (`venv` in this case)

    - Set up an environment by running `sh setup_env.sh`.
    - Run `sh run_pipeline.sh`.
