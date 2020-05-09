from src import fileManagement as fm
import os.path

if __name__ == '__main__':
    # Check if the user is connected to the network drive
    networkDir = fm.checkAccess()
    sensor = 'NogvaEngine'
    component = 'ME1' # optional

    year = 2019
    month = 11
    day = 6

    sl = fm.getSignalList('NogvaEngine', 'ME1')
    fileDir = fm.getSingleDayDir(networkDir, sensor, year, month, day)
    cols = fm.getSignalList(sensor, component)
    df = fm.concatenateFiles(fileDir, cols, index_col='time', chunksize=512)







    print("hello world!")

