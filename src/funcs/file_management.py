import datetime
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import win32file

from src.funcs import memory as mem


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

	if sys.platform == 'win32':
		print('Checking access to network drive..')
		DL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		drives = ['%s:' % d for d in DL if os.path.exists('%s:' % d)
			and win32file.GetDriveType('%s:' % d) == win32file.DRIVE_REMOTE]
		if(not drives):
			sys.exit('Error: access denied. Remember to connect to the network drive.')
		else:
			print('Access granted!')
			return drives[0]
	else:
		raise NotImplementedError(
								f"Operating system {sys.platform} is not " \
								"supported in the current implementation."
							)
		sys.exit()


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
					filter_operation=True,
					faulty_data=[]
				):
	"""Concatenate dataframes from imported csv.gz-format."""

	dfs = [] # list for storing dataframes to be concatenated

	# Get list of files to be concatenated from the desired file directory
	file_list = get_file_list(file_dir)

	# Filter data where the vessel is not in operation
	filter_column = 'ME1_EngineSpeed'
	filter_value = 1770 # speed when main engine 1 is in operation

	# Faulty data
	if faulty_data.__len__() != 0:
		faulty_dict = {}
		for interval in faulty_data:
			faulty_dict[interval[0]] = [interval[1], interval[2]]

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
		if faulty_data.__len__() != 0:
			if file in faulty_dict:
				simulated_error_start, simulated_error_end = faulty_dict[file][0], faulty_dict[file][1]
				df = remove_faulty_data(df, simulated_error_start, simulated_error_end, file)

		dfs.append(df) # append dataframe from file to the list of dataframes

	# If time is not specified as index column, ignore numbering indices in concatenated dataframe
	if index_col is None: ignore_index = True
	else: ignore_index = False # keep indexing if time is index column
	return concatenate_dataframes(dfs, index_col) # concatenate dataframes

def get_startpath(network_location,sensor=None,training_period=[None]*3):

	""""Return string with starting directory path based on desired data resolution."""
	[year, month, day] = training_period
	startpath = network_location
	for item in [sensor, year, month, day]:
		if item is not None: startpath = os.path.join(startpath, str(item))
		else: break
	return startpath

def get_data(
				root_dir,
				cols,
				index_col=None,
				chunksize=None,
				filter_operation=False,
				faulty_data=[]
			):

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
									filter_operation=filter_operation,
									faulty_data=faulty_data
								)

			dfs.append(df) # Append current dataframe to list of dataframes

	return concatenate_dataframes(dfs, index_col) # concatenate dataframes

def get_and_store_data(
						root_dir,
						cols,
						index_col=None,
						chunksize=None,
						filter_operation=False,
						file_suffix=None,
						faulty_data=[]
					):
	data = get_data(
						root_dir=root_dir,
						cols=cols,
						index_col=index_col,
						chunksize=chunksize,
						filter_operation=filter_operation,
						faulty_data=faulty_data
	)
	if data.empty:
		return False # ! REMOVE
		sys.exit(
				'No data in the selected interval qualifies the filtering ' \
				'conditions. No object has been pickled to memory.'
			)

	else:
		mem.store(data,file_suffix=file_suffix)
	return data

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
	"""Returns a datetime.time() object from a filename in known format."""

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

def remove_faulty_data(df, start, end, filename):
	"""Removes faulty data at a known time interval specified through a
	starting time and ending time. The input time values can either be in a
	known string format or as datetime.time() objects."""

	if(type(start) == str):
		start = to_datetime(start)
	if(type(end) == str):
		end = to_datetime(end)

	# Remove faulty data from df:
	# (U)sing end time before start time excludes the interval in-between)
	df_filtered = df.between_time(
		start_time=end,
		end_time=start,
		include_start=False,
		include_end=False
	)
	n_deleted_rows = df.shape[0] - df_filtered.shape[0]
	print(
		f"{n_deleted_rows} rows of faulty data between {start} and {end} " \
		f"succesfully removed from '{filename}'.")
	return df_filtered

def get_progress_bar(range_max, bar_desc=None):
	""""Returns an object representing a progress bar according to the
	progressbar module."""
	if bar_desc:
		print(f'{bar_desc}'.format())
	return progressbar.ProgressBar(maxval=range_max, \
		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


"""Parser to convert string object to datetime object when reading csv-files using pandas"""
date_parser = lambda time: pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f')

if __name__ == '__main__':
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
