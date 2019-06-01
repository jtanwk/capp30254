#!/bin/bash

module load python/3.6.1+intel-16.0
python3 -mvenv test_env
source ./test_env/bin/activate

module load python/3.6.1+intel-16.0
pip3 install --upgrade pip
pip3 install -r requirements.txt

deactivate
