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

def concatenateFiles(fileDir, cols=None, index_col=None,chunksize=None, filterOperation=True):
	"""Concatenate dataframes from imported csv.tz"""
	fileList = os.listdir(fileDir)

	# Filter data where the vessel is not in operation
	filterColumn = 'ME1_EngineSpeed'
	filterValue = 1770 # speed when main engine 1 is running

	li = []
	for file in fileList:
		df = pd.read_csv(fileDir + '\\' + file,
							compression='gzip',
							header=0,
							sep=',',
							usecols=cols,
							index_col=index_col,
							error_bad_lines=True)
		if(filterOperation):
			df = df[df[filterColumn] > filterValue]
		li.append(df)

	return pd.concat(li, axis=0, ignore_index = True)

def readSelectedColumnsFromFile(fileDir, cols):
	return pd.read_csv(fileDir, compression='gzip', header=0, sep=',', error_bad_lines=True)

# def list_files(startpath):
# 	sparseDegree = {}
# 	for root, dirs, files in os.walk(startpath):
# 		level = root.replace(startpath, '').count(os.sep)
# 		indent = ' ' * 4 * (level)
# 		print('{}{}/'.format(indent, os.path.basename(root)))
# 		subindent = ' ' * 4 * (level + 1)
# 		# for f in files:
# 		print('{}'.format(subindent))
# 		if files != []:
# 			df = concatenateFiles(root)
# 			sparseList = df.count()/df.shape[0]
# 			sparseDegree[getDatetime(files[0])] = allEqual(sparseList, 1)
# 	return sparseDegree
			# fileDir, fileList = getSingleDayFiles('Z',startpath, )

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

if __name__ == '__main__':
	# Meta = Meta()
	# checkPath()
	# sparseDegree = list_files('Z:nogvaEngine')
	# fileDir = getSingleDayDir('Z:','nogvaEngine',2019, 11, 21)
	# dataFrame = concatenateFiles(fileDir)
	# # fig = plt.line(dataFrame, x='time', y='ME1_ExhaustTemp1')
	# df = dataFrame.filter(['time','ME1_ExhaustTemp1', 'ME1_ExhaustTemp2'])
	# df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')
	# # 2019-11-21 10:50:00.000
	# df = df.set_index('time')
	# plt.figure()
	# df.plot()
	# plt.show()
	# dataFrame.count()/dataFrame.shape[0]
	# print('')

	dfs = []
	for d in range(1,8):
		dir = getSingleDayDir('Z:','NogvaEngine', 2020, 5, d)
		df = concatenateFiles(dir)
		df['time'] = toDatetime(df['time'])
		df = df.set_index('time')
		df = df[['ME1_ExhaustTemp1','ME1_ExhaustTemp2']]
		dfs.append(df)
	df = pd.concat(dfs)
	df.plot()
	plt.show()
	print('done')