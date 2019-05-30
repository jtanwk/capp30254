#!/bin/bash

source ./test_env/bin/activate

mkdir output
python3 hw5_pipeline.py

deactivate
