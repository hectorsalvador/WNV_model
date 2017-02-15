# WNV prediction model for the City of Chicago Department of Public Health
# Hector Salvador Lopez, Aug 2016

'''A script to process trap and weather data and obtain two databases:
an input csv for training ML models and an input csv for predictions'''

import argparse
import csv
import datetime
import pandas as pd
import sys, os
import sqlite3
from math import radians, degrees, cos, sin, asin, atan2, sqrt

###############################################################################
## Helper functions 														 ##
###############################################################################

def get_week(date):
	return datetime.datetime.strptime(date, '%Y-%m-%d').isocalendar()[1]

def find_centroid(coord_list):
	'''
	MODIFIED
	Followed instructions from: http://www.geomidpoint.com/calculation.html

	Inputs:
		coord_list, is a list of coordinates
	Returns:
		(lat, lon), a tuple indicating the weighted geographic center of the 
		given coordinates
	'''
	while None in coord_list:
		coord_list.remove(None)
	   
	if len(coord_list) == 0:
		return None

	weights = [1 for coord in coord_list]

	#convert lat and lon to radians
	lats = [radians(coord[0]) for coord in coord_list]
	lons = [radians(coord[1]) for coord in coord_list]

	# convert to cartesian coordinates
	x = [cos(lats[i]) * cos(lons[i]) for i in range(len(lats))]
	y = [cos(lats[i]) * sin(lons[i]) for i in range(len(lats))]
	z = [sin(lats[i]) for i in range(len(lats))]

	# compute weighted average
	xc = 0
	yc = 0
	zc = 0
	for i in range(len(weights)):
		xc += x[i] * weights[i]
		yc += y[i] * weights[i]
		zc += z[i] * weights[i]
	midpt = [xc / sum(weights), yc / sum(weights), zc / sum(weights)]

	# go back to degrees
	lon = degrees(atan2(yc, xc))
	hyp = sqrt(xc ** 2 + yc ** 2)
	lat = degrees(atan2(zc, hyp))

	return lat, lon

###############################################################################
## Helper functions for joining mosquito and weather data					 ##
###############################################################################

def initialize_obs(features):
	'''
	Initializes and returns a trap dictionary like:
	{'CULEX': 0, 'LOC': [], 'OTHER': 0, 'PIPIENS': 0, 'PRCP': 0, 'RESTUANS': 0,
	'TAVG': 0, 'TMAX': 0, 'TMIN': 0, 'WNVP': 0}
	'''
	d = {}
	for feat in features:
		d[feat] = 0
	d['LAT'] = []
	d['LON'] = []
	return d

def get_headers_and_index(csv_reader):
	'''
	Takes:
		- csv_reader, a csv.reader object that has read a csv file with mosquito
		trap data

	Returns:
		- headers, a list with the headers of the csv file
		- headers_index, a dictionary where keys are headers and values are their
		index 
	'''
	headers = csv_reader.__next__()
	headers_index = {}
	for i in range(len(headers)):
		headers_index[headers[i]] = i
	return headers, headers_index

def fix_coords(traps):
	for trap in traps.values():
		for week in trap.values():
			for year in week.values():
				if (len(year['LAT']) != 0) and (None not in year['LAT']):
					temp_list = [(year['LAT'][i], year['LON'][i]) for i \
								in range(len(year['LAT']))]
					year['LAT'], year['LON'] = find_centroid(temp_list)
				elif None in year['LAT']:
					year['LAT'] = ''
					year['LON'] = ''
		
def get_mosquito_features(line, headers_index):
	'''
	Takes:
		- line, list with info from a single observation from the mosquito trap
		data looking like:
			[1,2007-05-29,,CULEX PIPIENS/RESTUANS,41,N OAK PARK AVE,T002,
			41XX N OAK PARK AVE,41.956298856,-87.797517445,,1,0]
		- headers_index, a dictionary with indexes and header names of the
		mosquito trap data file

	Returns:
		- a dictionary with processed data from the line

	New variables collected from mosquito data should be specified here.
	'''
	def get(feat):
		return line[headers_index[feat]]
	trap  = get('Trap')
	ttype = get('TrapType')
	week  = int(get_week(get('Date')))
	year  = int(get('Date')[:4])
	wnvp  = int(get('WnvPresent'))
	try:
		lat = float(get('Latitude'))
		lon = float(get('Longitude'))
	except:
		lat = None
		lon = None
	species = get('Species')
	culex   = int(get('NumMosquitos'))
	pipiens, restuans, other = 0, 0, 0
	if   species == 'CULEX PIPIENS':
		pipiens  = culex
	elif species == 'CULEX RESTUANS':
		restuans = culex
	elif species == 'CULEX PIPIENS/RESTUANS': #maximum possible number of pipiens/restuans
		pipiens  = culex
		restuans = culex
	else:
		other = culex
	return {'TRAP': trap, 'WEEK': week, 'YEAR': year, 'WNVP': wnvp, 'LAT': lat,\
		'CULEX': culex, 'RESTUANS': restuans, 'PIPIENS': pipiens, 'OTHER': other, \
		'LON': lon, 'TYPE': ttype}

def fill_obs(traps_dict, line, headers_index, obs_features):
	'''
	Takes:
		- traps_dict, a dictionary with mosquito trap ids as keys
		- line, a list of values read from a csv line
		- headers_index, a dictionary with key, val pairs corresponding to 
			the column names of the mosquito file 
		- obs_features, values that traps_dict will have

	Modifies traps_dict in site without returning any value.
	New variables collected from mosquito data should be specified here.
	'''
	mosq_features = get_mosquito_features(line, headers_index)
	trap = mosq_features['TRAP']
	week = mosq_features['WEEK']
	year = mosq_features['YEAR']
	if trap not in traps_dict:
		traps_dict[trap] = {}
	if week not in traps_dict[trap]:
		traps_dict[trap][week] = {}
	if year not in traps_dict[trap][week]:
		traps_dict[trap][week][year] = initialize_obs(obs_features)
	d = traps_dict[trap][week][year]
	d['WNVP'] = max(d['WNVP'], mosq_features['WNVP'])
	d['LAT'].append(mosq_features['LAT'])
	d['LON'].append(mosq_features['LON'])
	d['TYPE'] = mosq_features['TYPE']
	mosq = ['CULEX', 'PIPIENS', 'RESTUANS', 'OTHER']
	for i in mosq:
		d[i] += mosq_features[i]

def fill_mosq(traps_dict, mosq_file, obs_features):
	'''
	Wrapper for fill_obs function.
	'''
	f = csv.reader(open(mosq_file))
	headers, headers_index = get_headers_and_index(f)
	for line in f:
		fill_obs(traps_dict, line, headers_index, obs_features)

def get_weather(wthr_db, weather_feats):
	'''
	Connects to a sqlite3 database to quickly fetch weather data.

	Takes:
		- wthr_db, a .db file with weather information
		- weather_feats, a list of weather features

	Returns:
		A dictionary of the form:
		rv = {'TMIN': { 2008: {33: 224.32,
								34: 228.12,
								...},
						2009: {22: 164.04,
								23: 182.11,
								...},
						...
						2016: {22: 164.04,
								23: 182.11,
								...},
						},
			  'TMAX': { 2008: { 33: 284.32,
								34: 268.12,
								...},
						2009: { 22: 198.04,
								23: 203.11,
								...},
						...
						2016: { 22: 200.04,
								23: 282.11,
								...},
						},...	
			}
	'''
	rv = {}
	conn = sqlite3.connect(wthr_db)
	c = conn.cursor()
	for feat in weather_feats:
		rv[feat] = {}
		c.execute('SELECT YEAR, WEEK, avg(' + feat + \
				') FROM data GROUP BY YEAR, WEEK')
		q_results = c.fetchall()
		for result in q_results:
			year = result[0]
			week = result[1]
			val  = result[2]
			if year not in rv[feat]:
				rv[feat][year] = {}
			if week not in rv[feat][year]:
				rv[feat][year][week] = {}
			rv[feat][year][week] = val
	c.close()

	# Add chronological index for the data
	rv['id'] = {}
	for result in q_results:
		year = result[0]
		week = result[1]
		val  = float("{}.{}".format(year, week))
		if year not in rv['id']:
				rv['id'][year] = {}
		if week not in rv['id'][year]:
			rv['id'][year][week] = {}
		rv['id'][year][week] = val
	
	return rv

###############################################################################
## Creating the new dataset													 ##
###############################################################################

def has_wnv_past_weeks(trap_dict, num_weeks, trap, week, year):
	'''
	Takes:
		- trap_dict, a dictionary of trap information
		- num_weeks, int indicating number of past weeks 
		- trap, a string indicating a trap (e.g. 'T002')
		- week, an int indicating a week number (e.g. 33)
		- year, an int indicating the year (e.g. 2012)

	Returns:
		- Number of times that a trap has had positive WNV occurrences
		in the past num_weeks
	'''
	s = [0]
	for i in range(num_weeks):
		if (week - i - 1 in trap_dict[trap].keys()) and (year in 
		trap_dict[trap][week - i - 1]):
			s.append(trap_dict[trap][week - i - 1][year]['WNVP'])
	return sum(s)

def times_wnv_this_week_any_past_year(trap_dict, trap, week):
	'''
	Takes:
		- trap_dict, a dictionary of trap information 
		- trap, a string indicating a trap (e.g. 'T002')
		- week, an int indicating a week number (e.g. 33)

	Returns:
		Number of times that this trap has had positive WNV occurrences
		in last years.
	'''
	if week not in trap_dict[trap].keys():
		return 0
	else:
		times = 0
		for year in trap_dict[trap][week]:
			times += trap_dict[trap][week][year]['WNVP']
		return times

def weather_past_weeks(weather_dict, num_weeks, week, year, weather_var):
	'''
	- weather_var can be any of ['AWND', 'TAVG', 'TMAX', 'TMIN', 'PRCP']
	'''
	s = []
	for i in range(num_weeks):
		if weather_dict[weather_var][year][week - i - 1] != None:
			s.append(weather_dict[weather_var][year][week - i - 1])
	return sum(s)

def count_mosq_past_weeks(trap_dict, num_weeks, trap, week, year, mosq_type):
	'''
	- mosq_type can be any of ['CULEX','PIPIENS','RESTUANS','OTHER']
	'''
	s = [0]
	for i in range(num_weeks):
		if (week - i - 1 in trap_dict[trap].keys()) and \
		(year in trap_dict[trap][week - i - 1].keys()):
			s.append(trap_dict[trap][week - i - 1][year][mosq_type])
	return sum(s)

def build_training_dataset(trap_dict, weather_dict, output):
	'''
	New variables can be added here
	'''
	db = []
	for trap in trap_dict.keys():
		for week in trap_dict[trap].keys():
			for year in trap_dict[trap][week].keys():
				CHRON = weather_dict['id'][year][week]
				TYPE  = trap_dict[trap][week][year]['TYPE']
				WNVW1 = has_wnv_past_weeks(trap_dict, 1, trap, week, year)
				WNVW2 = has_wnv_past_weeks(trap_dict, 2, trap, week, year)
				# WNVC1 =
				# WNVC2 =
				WNVHT = times_wnv_this_week_any_past_year(trap_dict, trap, week)
				WNVHW = times_wnv_this_week_any_past_year(trap_dict, trap, week + 1)
				CULX1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'CULEX')
				CULX2 = count_mosq_past_weeks(trap_dict, 2, trap, week, year, 'CULEX')
				PIPS1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'PIPIENS')
				PIPS2 = count_mosq_past_weeks(trap_dict, 2, trap, week, year, 'PIPIENS')
				REST1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'RESTUANS')
				REST2 = count_mosq_past_weeks(trap_dict, 2, trap, week, year, 'RESTUANS')
				OTHR1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'OTHER')
				OTHR2 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'OTHER')
				TAVG1 = weather_past_weeks(weather_dict, 1, week, year, 'TAVG')
				TMAX1 = weather_past_weeks(weather_dict, 1, week, year, 'TMAX')
				TMIN1 = weather_past_weeks(weather_dict, 1, week, year, 'TMIN')
				PRCP1 = weather_past_weeks(weather_dict, 1, week, year, 'PRCP')
				# WSF21 = weather_past_weeks(trap_dict, 1, trap, week, year, 'WSF2')
				# WSF51 = weather_past_weeks(trap_dict, 1, trap, week, year, 'WSF5')
				# TAVGN = 
				# TMAXN = 
				# TMINN =
				# PRCPN =
				LAT   = trap_dict[trap][week][year]['LAT']
				LON   = trap_dict[trap][week][year]['LON']
				WNVP = trap_dict[trap][week][year]['WNVP']
				db.append([trap, week, year, TYPE, CHRON, WNVW1, WNVW2, WNVHT,  \
					WNVHW, CULX1, CULX2, PIPS1, PIPS2, REST1, REST2, OTHR1,		\
					OTHR2, TAVG1, TMAX1, TMIN1, PRCP1, LAT, LON, WNVP])
	writer = csv.writer(open(output, "w"), lineterminator='\n')
	writer.writerow(['TRAP', 'WEEK', 'YEAR', 'TYPE', 'CHRON', 'WNVW1', 'WNVW2', \
		'WNVHT', 'WNVHW', 'CULX1', 'CULX2', 'PIPS1', 'PIPS2', 'REST1', 'REST2', \
		'OTHR1', 'OTHR2', 'TAVG1', 'TMAX1', 'TMIN1', 'PRCP1', 'LAT', 'LON',		\
		'WNVP'])
	writer.writerows(db)

def build_prediction_dataset(trap_dict, weather_dict, output, train_input_csv):
	'''
	'''
	def get_last_date(train_input_csv):
		df = pd.read_csv(train_input_csv)
		cron = list(set(df.CHRON.values))
		cron.sort()
		return str(cron[-1])
	
	FRIDAY = 5
	db = []
	last_date = get_last_date(train_input_csv) # get last day of data
	year, week = int(last_date[:4]), int(last_date[5:]) + 1
	print(year, week)

	for trap in trap_dict.keys():
		CHRON = eval(str(year) + '.' + str(week + 1))
		if week - 1 in trap_dict[trap].keys():
			if year in trap_dict[trap][week - 1].keys():
				TYPE  = trap_dict[trap][week - 1][year]['TYPE']
				WNVW1 = has_wnv_past_weeks(trap_dict, 1, trap, week, year)
				WNVW2 = has_wnv_past_weeks(trap_dict, 2, trap, week, year)
				# WNVC1 =
				# WNVC2 =
				WNVHT = times_wnv_this_week_any_past_year(trap_dict, trap, week)
				WNVHW = times_wnv_this_week_any_past_year(trap_dict, trap, week + 1)
				CULX1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'CULEX')
				CULX2 = count_mosq_past_weeks(trap_dict, 2, trap, week, year, 'CULEX')
				PIPS1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'PIPIENS')
				PIPS2 = count_mosq_past_weeks(trap_dict, 2, trap, week, year, 'PIPIENS')
				REST1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'RESTUANS')
				REST2 = count_mosq_past_weeks(trap_dict, 2, trap, week, year, 'RESTUANS')
				OTHR1 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'OTHER')
				OTHR2 = count_mosq_past_weeks(trap_dict, 1, trap, week, year, 'OTHER')
				TAVG1 = weather_past_weeks(weather_dict, 1, week, year, 'TAVG')
				TMAX1 = weather_past_weeks(weather_dict, 1, week, year, 'TMAX')
				TMIN1 = weather_past_weeks(weather_dict, 1, week, year, 'TMIN')
				PRCP1 = weather_past_weeks(weather_dict, 1, week, year, 'PRCP')
				# WSF21 = weather_past_weeks(trap_dict, 1, trap, week, year, 'WSF2')
				# WSF51 = weather_past_weeks(trap_dict, 1, trap, week, year, 'WSF5')
				# TAVGN = 
				# TMAXN = 
				# TMINN =
				# PRCPN =
				LAT   = trap_dict[trap][week - 1][year]['LAT']
				LON   = trap_dict[trap][week - 1][year]['LON']
				db.append([trap, week, year, TYPE, CHRON, WNVW1, WNVW2, WNVHT, \
						WNVHW, CULX1, CULX2, PIPS1, PIPS2, REST1, REST2, OTHR1,\
						OTHR2, TAVG1, TMAX1, TMIN1, PRCP1, LAT, LON])
				
	writer = csv.writer(open(output, "w"), lineterminator='\n')
	writer.writerow(['TRAP', 'WEEK', 'YEAR', 'TYPE', 'CHRON', 'WNVW1', \
		'WNVW2', 'WNVHT', 'WNVHW', 'CULX1', 'CULX2', 'PIPS1', 'PIPS2', \
		'REST1', 'REST2', 'OTHR1', 'OTHR2', 'TAVG1', 'TMAX1', 'TMIN1', \
		'PRCP1', 'LAT', 'LON'])
	writer.writerows(db)

###############################################################################
###############################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = \
		'Merge data from mosquito traps and weather databases.')
	parser.add_argument('--traps', required = False, 
		default = os.path.dirname(os.path.abspath(sys.argv[0])) + '/../data/traps.csv',
 		help = 'File with mosquito traps data.')
	parser.add_argument('--weather', required = False, 
		default= os.path.dirname(os.path.abspath(sys.argv[0])) + '/../data/weather.db',
 		help = 'File with weather data.')
	parser.add_argument('--train_o', required = False,
		default = os.path.dirname(os.path.abspath(sys.argv[0])) + '/../data/train_input.csv',
 		help = 'Path/Name of the output file for training inputs.')
	parser.add_argument('--pred_o', required = False,
		default = os.path.dirname(os.path.abspath(sys.argv[0])) + '/../data/pred_input.csv',
 		help = 'Path/Name of the output file for prediction inputs.')
	args = parser.parse_args()

	print('Merging {} and {} into {}.'.format(args.traps, args.weather, args.train_o))
	
	traps = {}
	mosquito_feats = ['WNVP', 'LAT', 'LON', 'TYPE', 'CULEX', 'RESTUANS', \
					  'PIPIENS', 'OTHER']
	weather_feats  = ['AWND', 'TAVG', 'TMAX', 'TMIN', 'PRCP']
	obs_features   = mosquito_feats + weather_feats

	fill_mosq(traps, args.traps, obs_features)
	fix_coords(traps)

	weather = get_weather(args.weather, weather_feats)

	# Create new database to feed into the models
	build_training_dataset(traps, weather, args.train_o)
	print('Training input data saved in {}.'.format(args.train_o))

	build_prediction_dataset(traps, weather, args.pred_o, args.train_o)
	print('Predictions input data saved in {}.'.format(args.pred_o))
	print('Done.')
