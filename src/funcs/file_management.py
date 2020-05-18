import datetime
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import win32file


def get_signal_list(sensor, component=None):
	"""Returns a parsed list of signals from specified columns in the
	columns.json file."""

	with open('columns.json', 'r') as f:
		json_columns = json.load(f)
	if not component:
		return list(json_columns['sensors'][sensor])
	else:
		return list(json_columns['sensors'][sensor][component])

def check_access():
	"""Check access to network drive with transmitted signal data. The function
	assumes that the user is only connected to one network drive."""

	print('Checking access to network drive..')
	DL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	drives = ['%s:' % d for d in DL if os.path.exists('%s:' % d)
            and win32file.GetDriveType('%s:' % d) == win32file.DRIVE_REMOTE]
	if(not drives):
		sys.exit('Error: access denied. Remember to connect to the network drive.')
	else:
		print('Access granted!')
		return drives[0]

def get_single_day_dir(network_location, sensor, year, month, date):
	"""Get the directory location for a desired sensor at a desired date"""

	return os.path.join(network_location, sensor, str(year), str(month).zfill(2), str(date).zfill(2))

def get_file_list(file_dir):
	try:
		return os.listdir(file_dir)
	except ValueError:
		print('Either a list of files or file directory must be provided.')
		sys.exit('Adjust user-specified input.')

def concatenate_files(file_dir,
					cols=None,
					index_col=None,
					chunksize=None,
					filter_operation=True
					):
	"""Concatenate dataframes from imported csv.gz-format."""

	dfs = [] # list for storing dataframes to be concatenated

	# Get list of files to be concatenated from the desired file directory
	file_list = get_file_list(file_dir)

	# Filter data where the vessel is not in operation
	filter_column = 'ME1_EngineSpeed'
	filter_value = 1770 # speed when main engine 1 is in operation

	# Iterate through files in desired file list
	for file in file_list:

		# Create a dataframe for current file with designated reading options
		df = pd.read_csv(file_dir + '\\' + file,
							compression='gzip',
							header=0,
							sep=',',
							usecols=cols,
							index_col=index_col,
							parse_dates=['time'],
							date_parser=date_parser,
							chunksize=chunksize,
							error_bad_lines=True)

		# Apply operation filter if specified by function input
		if(filter_operation):
			df = df[df[filter_column] > filter_value] # filter based on appropriate measures


		# Remove simulated error induced 21-nov 2019 between 10:50:16 and 10:56:33
		if file == 'NogvaEngine_20191121_105000.csv.gz':
			simulated_error_start = '2019-11-21 10:50:16.000'
			simulated_error_end =  '2019-11-21 10:56:33.000'
			df = remove_faulty_data(df, simulated_error_start, simulated_error_end)

		dfs.append(df) # append dataframe from file to the list of dataframes

	# If time is not specified as index column, ignore numbering indices in concatenated dataframe
	if index_col is None: ignore_index = True
	else: ignore_index = False # keep indexing if time is index column
	return concatenate_dataframes(dfs, index_col) # concatenate dataframes

def get_startpath(network_location,sensor=None,year=None,month=None,day=None):

	""""Return string with starting directory path based on desired data resolution."""
	startpath = network_location
	for item in [sensor, year, month, day]:
		if item is not None: startpath = os.path.join(startpath, str(item))
		else: break
	return startpath

def get_data(root_dir,cols,index_col=None,chunksize=None,filter_operation=False):

	"""Retrieves a concatenated dataframe with all data from the gives root directory."""

	dfs = [] # list for storing dataframes to be concatenated

	# Iterate through file hierarchy based on the given root directory
	for root, dirs, files in os.walk(root_dir):

		# If list of files at current directory is not empty, concatenate containing files
		if files != []:
			print(root) # ! REMOVE
			# Concatenate files in the given directory based on specified options
			df = concatenate_files(file_dir=root,
									cols=cols,
									index_col=index_col,
									chunksize=chunksize,
									filter_operation=filter_operation)

			dfs.append(df) # Append current dataframe to list of dataframes

	return concatenate_dataframes(dfs, index_col) # concatenate dataframes

def concatenate_dataframes(dfs,index_col=None):
	"""Concatenate list of dataframes. If time is not specified as index
	column through the index_col variable, the indexed numbering is ignored in
	the concatenated dataframe."""

	if index_col is None: ignore_index = True
	else: ignore_index = False
	return pd.concat(dfs,axis=0,ignore_index=ignore_index) # return concatenated dataframe

def all_equal(list, val):
	"""Returns boolean value corresponding to all values in a list array being equal"""

	return all(elem == val for elem in list)

def get_datetime(filename):
	"""Returns a datetime.time() object from a file name in known format."""

	time_string = filename[:filename.rfind('_')]
	time_string = time_string[time_string.rfind('_')+1:]
	year = int(time_string[0:4])
	month = int(time_string[4:6])
	day = int(time_string[6:])
	return datetime.datetime(year=year, month=month, day=day).time()

def to_datetime(time_string, format='%Y-%m-%d %H:%M:%S.%f'):
	"""Returns a datetime.time() object from a string in known format."""

	return datetime.datetime.strptime(time_string, format).time()

def df_plot(df, values):
	"""Plots a dataframe through the matplotlib library."""

	df.filter(values).plot()
	plt.show()

def remove_faulty_data(df, start, end):
	"""Removes faulty data at a known time interval specified through a
	starting time and ending time. The input time values can either be in a
	known string format or as datetime.time() objects."""

	if(type(start) == str):
		start = to_datetime(start)
	if(type(end) == str):
		end = to_datetime(end)
	return df.between_time(start_time=end,end_time=start,include_start=False,include_end=False) # end time before start time to exclude the interval in-between

def get_progress_bar(range_max, bar_desc=None):
	""""Returns an object representing a progress bar according to the
	progressbar module."""
	if bar_desc:
		print('{}'.format(bar_desc))
	return progressbar.ProgressBar(maxval=range_max, \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


"""Parser to convert string object to datetime object when reading csv-files using pandas"""
date_parser = lambda time: pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f')

if __name__ == '__main__':
	sys.exit('Run from manage.py, not file management.')
