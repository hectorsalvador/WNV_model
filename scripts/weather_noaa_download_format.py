###########################################################################
## WNV prediction model for the City of Chicago Department of Public Health
## Hector Salvador Lopez, Aug 2016
###########################################################################

import argparse
import csv
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta 
import requests
import json

###############################################################################
## Helper functions													 		 ##
###############################################################################

def url_date_intervals(start, end):
	'''
	Takes:
		- start, string representing the starting date to retreive data YYYY-MM-DD
		- end, string representing the starting date to retreive data YYYY-MM-DD

	Returns:
		- rv, a list with the dates to input in a NOAA API url request
	'''
	rv = []
	start = datetime.strptime(start, "%Y-%m-%d").date()
	end = datetime.strptime(end, "%Y-%m-%d").date()
	new_date = start + relativedelta(years=1)

	date = datetime.strftime(start, "%Y-%m-%d").replace(' 0', ' ')
	rv.append(date)

	while new_date < end:
		# print(new_date, end)
		date = datetime.strftime(new_date, "%Y-%m-%d").replace(' 0', ' ')
		rv.append(date)
		new_date = new_date + relativedelta(years=1)

	# print(end)
	end = datetime.strftime(end, "%Y-%m-%d").replace(' 0', ' ')
	rv.append(end)

	return rv

def store(dict, name):
	with open(name, 'w') as f:
	    json.dump(dict, f, sort_keys=True, indent=4)


###############################################################################
## URL request functions													 ##
###############################################################################

def build_url(datasetid, stationid, startdate, enddate, limit, offset = 0):
	'''
	Builds urls like:
		http://www.ncdc.noaa.gov/cdo-web/api/v2/data?
		datasetid=GHCND&
		stationid=GHCND:USW00094846&
		startdate=2015-01-01&
		enddate=2016-01-01&
		limit=1000'
	
	These urls are used to retrieve monthly weather information from the NOAA API.
	http://www.ncdc.noaa.gov/cdo-web/webservices/v2
	'''
	url = 'http://www.ncdc.noaa.gov/cdo-web/api/v2/data?'
	did = 'datasetid={}'.format(datasetid)
	sid = 'stationid={}'.format(stationid)
	# stationid = 'stationid=GHCND:USW00014819' 
	sdt = 'startdate={}'.format(startdate)
	edt = 'enddate={}'.format(enddate)
	lim = 'limit={}'.format(limit)
	off = 'offset={}'.format(offset)
	if offset != 0:
		url = url + '&'.join([did, sid, sdt, edt, lim, off])
	else:
		url = url + '&'.join([did, sid, sdt, edt, lim])

	return url

def one_request(url, token):
	'''
	Gets weather data from a request to the NOAA API.
	Takes:
		- url, a string containing the url request
		- token, a string containing a token that allows to make requests to 
		the NOAA API
	
	Returns:
		- weather, a json file with the weather results
	'''	
	r = requests.get(url, headers = token)	
	assert r.ok, 'Error {} with following URL:\n{}.'.format(r.status_code, r.url)
	weather = r.json()['results']
	r.close()
	return weather

def multi_request(db, station, start, end, limit, token, verbose = False):
	'''
	For requests that return exactly 1000 results, repeat the request to get 
	more weather results.
	'''
	url = build_url(db, station, start, end, limit)
	if verbose:
		print('Getting weather from NOAA API. Dates: {} to {}.'.format(start, end))
	weather = one_request(url, token)
	i = 0
	while len(weather) % MAX_RES == 0:
		i += 1
		new_url = build_url(db, station, start, end, limit, i * MAX_RES)
		w_temp  = one_request(new_url, token)
		weather = weather + w_temp
	return weather

###############################################################################
## Format weather files functions											 ##
###############################################################################

def init_date_dict(headers):
	'''
	Initializes a dictionary where keys are values from headers.
	Takes:
		- headers, a list of headers

	Returns:
		- rv, a dictionary
	'''
	rv = {}
	for val in headers:
		rv.setdefault(val, None)
	return rv

def from_list_to_dict(w_list, headers):
	'''
	Convert a list of dictionaries looking like this:
		[...{'attributes': 'H,,S,',
	  	'datatype': 'TAVG',
	  	'date': '2015-03-23T00:00:00',
	  	'station': 'GHCND:USW00094846',
	  	'value': -16},
	 	{'attributes': ',,W,2400',
	  	'datatype': 'TMAX',
	  	'date': '2015-03-23T00:00:00',
	  	'station': 'GHCND:USW00094846',
	  	'value': -5},
	 	{'attributes': ',,W,2400',
	  	'datatype': 'TMIN',
	  	'date': '2015-03-23T00:00:00',
	  	'station': 'GHCND:USW00094846',
	  	'value': -21},...]

  	Into a dictionary looking like:
	  	{'2015-03-23': {'WEEK':  12, 
	  					'TAVG': -16, 
	  					'TMAX':  -5, 
	  					'TMIN': -21,
	  					...}, 
	  	...}

	Takes:
		- w_list, a list of dictionaries with weather data
		- headers, a list of strings with the NOAA weather types (e.g. 'TMIN',
		'PRCP')

	Returns:
		- rv, a dictionary of dictionaries with weather data
	'''
	rv = {}
	for elem in w_list:
		date  = elem['date'][:10]
		week  = datetime.strptime(date, '%Y-%m-%d').isocalendar()[1]
		year  = elem['date'][:4]
		dtype = elem['datatype']
		val   = elem['value']
		if date not in rv:
			rv[date] = init_date_dict(headers)
			rv[date]['WEEK'] = week
			rv[date]['YEAR'] = year
		rv[date][dtype] = val
	return rv

def from_dict_to_csv(w_dict, headers, csv_name):
	'''
	Takes a dictionary looking like this:

	  	{'2015-03-23': {'WEEK':  12, 
	  					'TAVG': -16, 
	  					'TMAX': -5, 
	  					'TMIN': -21,
	  					...}, 
	  	...}

	And converts it to a csv looking like this:

		'DATE','WEEK',AWND','PRCP','SNOW','SNWD', ...
		'2015-03-23',14,-16,-5,-21,...
		'2015-03-24',14,-16,-4,-20,...

	Takes:
		- w_dict, a dictionary of weather data
		- headers, a list of strings with weather types
		- csv_name, a string with the name of the csv output
	'''
	f = csv.writer(open(csv_name, "w"))
	f.writerow(['DATE'] + headers)
	for date, vals in w_dict.items():
		f.writerow([date] + [vals[i] for i in headers])

###############################################################################
## Wrapper																	 ##
###############################################################################

MAX_RES = 1000 		# Maximum results returned by the NOAA API.

def go(db, station, start, end, limit, token, headers, output, verbose = False):
	'''
	Wrapper function.
	'''
	# In case debugging is required or API calls are depleted
	json_name =  'data/weather' + start + '_' + end + '.json' 

	weather = []
	date_intervals = url_date_intervals(start, end)
	for i in range(len(date_intervals) - 1):
		start_temp = date_intervals[i]
		end_temp   = date_intervals[i + 1]
		weather    = weather + multi_request(db, station, start_temp, end_temp, \
					limit, token, True)
	if verbose:
		print('Request successful.')

	store(weather, json_name)

	if verbose:
		print('Formatting data.')
		
	weather_dict = from_list_to_dict(weather, headers)
	from_dict_to_csv(weather_dict, headers, output) 
	print('Data saved to {}'.format(output))

###############################################################################
###############################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Download and format weather data.')
	parser.add_argument('--db', default = 'GHCND', help = 'NOAA database to access.')
	parser.add_argument('--station', default= 'GHCND:USW00094846', \
		help = 'Station with relevant data.')
	parser.add_argument('--start', default = '2007-01-01', help = 'First day to retrieve.')
	parser.add_argument('--end', default = '2016-01-01', help = 'Last day to retrieve.')
	parser.add_argument('--limit', default = 1000, \
		help = 'Maximum number of results to retrieve.')
	parser.add_argument('--token', required = True, help = 'NOAA API personal token.')
	parser.add_argument('--output', default = 'test.csv', help = 'The output csv file.')
	args = parser.parse_args()

	token = {'token':'{}'.format(args.token)}
	HEADERS = ['YEAR', 'WEEK', 'AWND', 'PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX', \
		'TMIN', 'WDF2', 'WDF5', 'WSF2', 'WSF5', 'WT01', 'WT02', 'WT04', 'WT06',\
		'WT08', 'WT09']

	go(args.db, args.station, args.start, args.end, args.limit, token,\
		HEADERS, args.output)
