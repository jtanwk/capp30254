#!/bin/bash

python3 -mvenv test_env
source ./test_env/bin/activate

pip3 install --user --upgrade pip
pip3 install --user -r requirements.txt

deactivate
