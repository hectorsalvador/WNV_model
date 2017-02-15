#!/bin/bash

PERIODS=$1

# Run model
if [ -z $PERIODS ]
then 
	echo "Training model using defaults."
	python3 -W ignore scripts/build_model.py
else
	echo "Training model every $PERIODS periods."
	python3 -W ignore scripts/build_model.py --p $PERIODS
fi
