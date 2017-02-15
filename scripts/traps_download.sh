#!/bin/bash

# It will always be assumed this script is called by download_input_data.sh from root dir

# Define day, url, and csv filename variables
WEBPAGE="https://data.cityofchicago.org/api/views/jqe8-8r6s/rows.csv?accessType=DOWNLOAD"
FILE=$1
FILE_SORTED="$1_sorted"
OUTPUT=$2

echo 
echo "Downloading $FILE.."
wget -O $FILE $WEBPAGE
echo

echo "Sorting $FILE into $FILE_SORTED"
awk 'NR > 1' $FILE | sort > $FILE_SORTED
HEADER=$(awk 'NR==1' $FILE)
sed -i "1s/^/$HEADER\n/" $FILE_SORTED
echo

echo "Formatting $FILE_SORTED and saving to $OUTPUT."
python scripts/traps_format.py --input $FILE_SORTED --output $OUTPUT
echo "Done"
echo
