# WNV prediction model for the City of Chicago Department of Public Health
# Hector Salvador Lopez, Aug 2016

'''Formats data downloaded from the '''

import argparse
import csv
import numpy as np
import re

def format_data(i_filename, o_filename):
	lis = []
	i = 1
	with open(i_filename) as f:
		weather_head = f.__next__()
		for line in csv.reader(f):
			date       = line[6]
			ymd        = date[6:10]+'-'+date[:2]+'-'+date[3:5]
			species    = line[9]
			address    = line[3]
			if address != '':
				block  = int(re.search('[0-9]*', address).group(0))
			else:
				block  = ''
			if address != '':
				street = re.search('(XX )([A-Z ]+)', address).group(2)
			else:
				street = ''
			trap       = line[4]
			trap_type  = line[5]
			street_add = line[3]
			latitude   = line[10]
			longitude  = line[11]
			mosquitos  = line[7]
			if line[8] == 'negative':
				wnv = 0
			else:
				wnv = 1
			lis.append([i, ymd, species, block, street, trap, trap_type,\
						street_add, latitude, longitude, mosquitos, wnv])
			i += 1

	with open(o_filename, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		header = ["Id", "Date", "Species", "Block", "Street", "Trap", \
		"TrapType", "AddressNumberAndStreet", "Latitude", "Longitude",\
		"NumMosquitos", "WnvPresent"]
		writer.writerow(header)
		for val in lis:
			writer.writerow(val)

###############################################################################
###############################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Format WNV data.')
	parser.add_argument('--input', required=True, help='The csv file to format.')
	parser.add_argument('--output', required=True, help='The output csv file.')

	args = parser.parse_args()

	format_data(args.input, args.output)
