import pandas as pd
import os
import time, datetime
import matplotlib.pyplot as plt
import win32file
import sys
import json

def getSignalList(sensor, component=None):
	with open('columns.json', 'r') as f:
		jsonColumns = json.load(f)
	if not component:
		return list(jsonColumns['sensors'][sensor])
	else:
		return list(jsonColumns['sensors'][sensor][component])

def checkAccess():
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

def concatenateFiles(fileList=None,
						fileDir=None,
						cols=None,
						indexCol='time',
						chunksize=None,
						filterOperation=True
					):
	"""Concatenate dataframes from imported csv.gz-format."""

	if not fileList:
		try:
			fileList = os.listdir(fileDir)
		except ValueError:
			print('Either a list of files or file directory must be provided.')
			sys.exit('Adjust user-specified input.')

	# Filter data where the vessel is not in operation
	filterColumn = 'ME1_EngineSpeed'
	filterValue = 1770 # speed when main engine 1 is in operation

	dfs = []
	for file in fileList:
		df = pd.read_csv(fileDir + '\\' + file,
							compression='gzip',
							header=0,
							sep=',',
							usecols=cols,
							index_col=indexCol,
							chunksize=chunksize,
							error_bad_lines=True)
		if(filterOperation):
			df = df[df[filterColumn] > filterValue]
		dfs.append(df)
	if indexCol is None: ignoreIndex = True
	else: ignoreIndex = False
	return pd.concat(dfs,axis=0,ignore_index=ignoreIndex)

def getStartpath(networkLocation,sensor=None,year=None,month=None,day=None):
	startpath = networkLocation
	for item in [sensor, year, month, day]:
		if item is not None: startpath = os.path.join(startpath, str(item))
		else: break
	return startpath



"""
fm.getData(
                        fileDir=startpath,
                        cols=cols,
                        indexCol=indexCol,
                        chunksize=chunksize,
                        filterOperation=filterOperation
                    )
"""

def getData(rootDir,cols,indexCol=None,chunksize=None,filterOperation=True):
	if indexCol is None: ignoreIndex = True
	else: ignoreIndex = False

	dfs = []
	for root, dirs, files in os.walk(rootDir):
		if files != []:
			print(root)
			df = concatenateFiles(fileDir=root,
									cols=cols,
									indexCol=indexCol,
									chunksize=chunksize,
									filterOperation=filterOperation)
			print(df.shape)
			dfs.append(df)
	return pd.concat(dfs,axis=0,ignore_index=ignoreIndex)

def allEqual(list, val):
	"""Returns boolean value corresponding to all values in a list array being equal"""
	return all(elem == val for elem in list)

def getDatetime(fileName):
	timeString = fileName[:fileName.rfind('_')]
	timeString = timeString[timeString.rfind('_')+1:]
	year = int(timeString[0:4])
	month = int(timeString[4:6])
	day = int(timeString[6:])
	return datetime.datetime(year=year, month=month, day=day)

def toDatetime(time, format='%Y-%m-%d %H:%M:%S.%f'):
	return pd.to_datetime(time, format=format)

def plotDataframe(df):
	df.plot()
	plt.show()

if __name__ == '__main__':
	sys.exit('Run from manage-file, not file management.')