from src import fileManagement as fm

if __name__ == '__main__':
    # Check if the user is connected to the network drive
    networkDir = fm.checkAccess()

    ##########################################################################
    ############################### INPUT DATA ###############################
    # Desired sensor(s) and component(s)
    sensors = 'NogvaEngine'
    components = 'ME1'      # optional
    # components = {'NogvaEngine':['ME1', 'ME2']
    #             }

    # Desired time-period of training set
    year = 2019     # None: all available data will be used
    month = None    # None: all available data in given year will be used
    day = None      # None: all available data in given month will be used

    # Reading and filtering of data
    indexCol = 'time' # index column
    chunksize = None # None: the model will train load all data simultaneously
    filterOperation = True  # If True, only in-operation data will be used

    # Filter condition (problem: hard to define logical operator here)
    # filterColumn = 'ME1_EngineSpeed'
    # filterValue = 1770

    ##########################################################################
    ##########################################################################

    # Get all signals from desired sensor(s) and component(s)
    cols = fm.getSignalList(sensors, components)

    # Starting path based on desired sensors or time periods
    startpath = fm.getStartpath(networkDir, sensors, year, month, day)

    # Get a dataframe containing desired data in desired formats
    data = fm.getData(
                        rootDir=startpath,
                        cols=cols,
                        indexCol=indexCol,
                        chunksize=chunksize,
                        filterOperation=filterOperation
                    )
    dfExhaustTempME1 = data.filter(['ME1_ExhaustTemp1','ME1_ExhaustTemp2'])
    dfExhaustTempME1.plot()
    plt.show()
    print("hello world!")