#!/bin/bash

# install virtualenv and activate it 
# for more information visit https://virtualenv.pypa.io/en/stable/userguide/
VENV="venv"
pip3 install virtualenv
virtualenv -p python3 $VENV
echo
echo "Activate virtual environment to continue."
