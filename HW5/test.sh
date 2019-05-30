#!/bin/bash

python -mvenv test_env
source ./test_env/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt

python3 hw5_pipeline.py

deactivate
