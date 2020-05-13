from src import fileManagement as fm
from src import memory as mem
from models import sample as model

if __name__ == '__main__':
    # Check if the user is connected to the network drive
    networkDir = fm.checkAccess()

    ##########################################################################
    ###########################     INPUT DATA     ###########################
    ##########################################################################
    # Desired sensor(s) and component(s)
    sensors = 'NogvaEngine'
    components = 'ME1'      # optional
    # components = {'NogvaEngine':['ME1', 'ME2']
    #             }

    # Desired time-period of training set
    year = 2019  # None: all available data will be used
    month = 11 # None: all available data in given year will be used
    day = 21 # None: all available data in given month will be used

    # Reading and filtering of data
    indexCol = 'time' # index column
    chunksize = None # None: the model will train load all data simultaneously
    filterOperation = True  # If True, only in-operation data will be used

    # Filter condition (problem: hard to define logical operator here)
    # filterColumn = 'ME1_EngineSpeed'
    # filterValue = 1770

    ##########################################################################
    ###########################  RETRIEVING DATA   ###########################
    ##########################################################################

    # Get all signals from desired sensor(s) and component(s)
    cols = fm.getSignalList(sensors, components)

    # Starting path based on desired sensors or time periods
    startpath = fm.getStartpath(networkDir, sensors, year, month, day)

    # Get a dataframe containing desired data in desired formats

    createDataFile = False
    if createDataFile:
        data = fm.getData(
                            rootDir=startpath,
                            cols=cols,
                            indexCol=indexCol,
                            chunksize=chunksize,
                            filterOperation=filterOperation
                        )
        if data.empty:
            print('No data in the selected interval qualifies the filtering conditions. No variables have been pickled to memory.')
        else:
            filenameStoring = 'store'
            mem.store(data, filenameStoring)
            mem.storeTimeInterval(data.index[0], data.index[-1], filenameStoring + '_timeint')

    ##########################################################################
    ###########################   CREATING MODEL  ############################
    ##########################################################################
    filenameLoading = 'store_nov_2019'
    data = mem.load(filenameLoading)
    timeint = mem.loadMeta(filenameLoading + '_timeint')
    print('Data from {} to {} loaded into memory.'.format(timeint[0], timeint[-1]))

    model.create(data)

    ##########################################################################
    ########################### VISUALIZE RESULTS ############################
    ##########################################################################
    values = ['ME1_ExhaustTemp1','ME1_ExhaustTemp2']
    fm.dfPlot(data, values)


    print("Hello world!") # ! REMOVE (used for debugging breakpoint)