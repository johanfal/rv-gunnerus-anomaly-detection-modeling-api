import pandas as pd
import os
import time, datetime
import matplotlib.pyplot as plt
import win32file
import sys
import json

def getSignalList(sensor, component=None):
	"""Returns a parsed list of signals from specified columns in the
	columns.json file."""

	with open('columns.json', 'r') as f:
		jsonColumns = json.load(f)
	if not component:
		return list(jsonColumns['sensors'][sensor])
	else:
		return list(jsonColumns['sensors'][sensor][component])

def checkAccess():
	"""Check access to network drive with transmitted signal data. The function
	assumes that the user is only connected to one network drive."""

	print('Checking access to network drive..')
	dl = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	drives = ['%s:' % d for d in dl if os.path.exists('%s:' % d)
            and win32file.GetDriveType('%s:' % d) == win32file.DRIVE_REMOTE]
	if(not drives):
		sys.exit('Error: access denied. Remember to connect to the network drive.')
	else:
		print('Access granted!')
		return drives[0]

def getSingleDayDir(networkLocation, sensor, year, month, date):
	"""Get the directory location for a desired sensor at a desired date"""

	dir = os.path.join(networkLocation, sensor, str(year), str(month).zfill(2), str(date).zfill(2))
	return dir

def getFileList(fileDir):
	try:
		return os.listdir(fileDir)
	except ValueError:
		print('Either a list of files or file directory must be provided.')
		sys.exit('Adjust user-specified input.')

def concatenateFiles(fileDir,
					cols=None,
					indexCol=None,
					chunksize=None,
					filterOperation=True
					):
	"""Concatenate dataframes from imported csv.gz-format."""

	dfs = [] # list for storing dataframes to be concatenated

	# Get list of files to be concatenated from the desired file directory
	fileList = getFileList(fileDir)

	# Filter data where the vessel is not in operation
	filterColumn = 'ME1_EngineSpeed'
	filterValue = 1770 # speed when main engine 1 is in operation

	# Iterate through files in desired file list
	for file in fileList:

		# Create a dataframe for current file with designated reading options
		df = pd.read_csv(fileDir + '\\' + file,
							compression='gzip',
							header=0,
							sep=',',
							usecols=cols,
							index_col=indexCol,
							parse_dates=['time'],
							date_parser=dateParser,
							chunksize=chunksize,
							error_bad_lines=True)

		# Apply operation filter if specified by function input
		if(filterOperation):
			df = df[df[filterColumn] > filterValue] # filter based on appropriate measures


		# Remove simulated error induced 21-nov 2019 between 10:50:16 and 10:56:33
		if file == 'NogvaEngine_20191121_105000.csv.gz':
			simulatedErrorStart = '2019-11-21 10:50:16.000'
			simulatedErrorEnd =  '2019-11-21 10:56:33.000'
			df = removeFaultyData(df, simulatedErrorStart, simulatedErrorEnd)

		dfs.append(df) # append dataframe from file to the list of dataframes

	# If time is not specified as index column, ignore numbering indices in concatenated dataframe
	if indexCol is None: ignoreIndex = True
	else: ignoreIndex = False # keep indexing if time is index column
	return concatenateDataframes(dfs, indexCol) # concatenate dataframes

def getStartpath(networkLocation,sensor=None,year=None,month=None,day=None):

	""""Return string with starting directory path based on desired data resolution."""
	startpath = networkLocation
	for item in [sensor, year, month, day]:
		if item is not None: startpath = os.path.join(startpath, str(item))
		else: break
	return startpath

def getData(rootDir,cols,indexCol=None,chunksize=None,filterOperation=False):

	"""Retrieves a concatenated dataframe with all data from the gives root directory."""

	dfs = [] # list for storing dataframes to be concatenated

	# Iterate through file hierarchy based on the given root directory
	for root, dirs, files in os.walk(rootDir):

		# If list of files at current directory is not empty, concatenate containing files
		if files != []:
			print(root) # ! REMOVE
			# Concatenate files in the given directory based on specified options
			df = concatenateFiles(fileDir=root,
									cols=cols,
									indexCol=indexCol,
									chunksize=chunksize,
									filterOperation=filterOperation)

			dfs.append(df) # Append current dataframe to list of dataframes

	return concatenateDataframes(dfs, indexCol) # concatenate dataframes

def concatenateDataframes(dfs,indexCol=None):
	"""Concatenate list of dataframes. If time is not specified as index
	column through the indexCol variable, the indexed numbering is ignored in
	the concatenated dataframe."""

	if indexCol is None: ignoreIndex = True
	else: ignoreIndex = False
	return pd.concat(dfs,axis=0,ignore_index=ignoreIndex) # return concatenated dataframe

def allEqual(list, val):
	"""Returns boolean value corresponding to all values in a list array being equal"""

	return all(elem == val for elem in list)

def getDatetime(fileName):
	"""Returns a datetime.time() object from a file name in known format."""

	timeString = fileName[:fileName.rfind('_')]
	timeString = timeString[timeString.rfind('_')+1:]
	year = int(timeString[0:4])
	month = int(timeString[4:6])
	day = int(timeString[6:])
	return datetime.datetime(year=year, month=month, day=day).time()

def toDatetime(timeString, format='%Y-%m-%d %H:%M:%S.%f'):
	"""Returns a datetime.time() object from a string in known format."""

	return datetime.datetime.strptime(timeString, format).time()

def dfPlot(df, values):
	"""Plots a dataframe through the matplotlib library."""

	df.filter(values).plot()
	plt.show()

def removeFaultyData(df, start, end):
	"""Removes faulty data at a known time interval specified through a
	starting time and ending time. The input time values can either be in a
	known string format or as datetime.time() objects."""

	if(type(start) == str):
		start = toDatetime(start)
	if(type(end) == str):
		end = toDatetime(end)
	return df.between_time(start_time=end,end_time=start,include_start=False,include_end=False) # end time before start time to exclude the interval in-between

"""Parser to convert string object to datetime object when reading csv-files using pandas"""
dateParser = lambda time: pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f')

if __name__ == '__main__':
	sys.exit('Run from manage.py, not file management.')