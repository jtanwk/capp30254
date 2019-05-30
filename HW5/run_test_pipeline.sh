#!/bin/bash

source ./test_env/bin/activate

mkdir output
python3 test_pipeline.py

deactivate
