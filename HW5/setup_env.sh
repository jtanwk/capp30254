#!/bin/bash

python3 -mvenv test_env
source ./test_env/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt

deactivate
