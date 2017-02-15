#!/bin/sh

# It will always be assumed this script is called by download_input_data.sh from root dir

# Define day, url, and csv filename variables
DAY=$(date +"%F")
OUTPUT=$1
TOKEN=$(cat scripts/weather_noaa_token.txt)

# Download data
echo 
echo "Checking if $OUTPUT exists in data directory."
if [ ! -f $OUTPUT ]
	then
		echo "  $OUTPUT not found.."
		echo "  Downloading $OUTPUT.."
		python scripts/weather_noaa_download_format.py --start '2007-01-01' --end $DAY  --token $TOKEN --output $OUTPUT
fi
echo "Download complete."
echo
