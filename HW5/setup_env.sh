#!/bin/bash

python -mvenv test_env
source ./test_env/bin/activate

module load python/3.6.1+intel-16.0
pip3 install --user --upgrade pip
pip3 install --user -r requirements.txt

deactivate
