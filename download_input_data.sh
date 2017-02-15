#!/bin/bash

## Update mosquito traps and weather data
DAY=$(date +"%F")
TRAP_DWLD="data/trapdata_$DAY.csv"
TRAP_OUTPUT="data/traps.csv"
WEATHER_DWLD="data/weather_noaa_$DAY.csv"
WEATHER_OUTPUT="data/weather.db"
TRAIN_O="data/train_input.csv"
PRED_O="data/pred_input.csv"

## Obtain results of trap data from data.portal (if the traps have WNV present)
sh ./scripts/traps_download.sh $TRAP_DWLD $TRAP_OUTPUT

## Obtain weather data
sh ./scripts/weather_noaa_download_format.sh $WEATHER_DWLD

## Put sorted weather into database file
echo "Dumping data/weather_sorted.csv into a data/weather.db."
python3 scripts/csv2sqlite.py $WEATHER_DWLD $WEATHER_OUTPUT
echo "Done."

## Process and merge data
python3 scripts/process_input_data.py --traps $TRAP_OUTPUT --weather $WEATHER_OUTPUT --train_o $TRAIN_O --pred_o $PRED_O
