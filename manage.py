# Custom module imports
from src.funcs import file_management as fm
from src.funcs import memory as mem
from src.models import lstm_sample as model

if __name__ == '__main__':
    # Check if the user is connected to the network drive
    network_dir = fm.check_access()

    ##########################################################################
    ###########################     INPUT DATA     ###########################
    ##########################################################################
    # Desired sensor(s) and component(s)
    SENSORS = 'NogvaEngine'
    COMPONENTS = 'ME1' # optional
    PREDICTION_COLS = ['ME1_ExhaustTemp1','ME1_ExhaustTemp2'] # desired columns to predict the values of. If None, all values will be predicted

    # COMPONENTS = {'NogvaEngine':['ME1', 'ME2']
    #             }

    # Desired time-period of training set
    # § IMPROVEMENT: should be able to select range of years, months, or days
    YEAR = None  # None: all available data will be used
    MONTH = None # None: all available data in given year will be used
    DAY = 21 # None: all available data in given month will be used

    # Reading and filtering of data
    INDEX_COL = 'time' # index column
    CHUNKSIZE = None # None: the model will train load all data simultaneously
    FILTER_OPERATION = True  # If True, only in-operation data will be used

    # Filter condition (problem: hard to define logical operator here)
    # filterColumn = 'ME1_EngineSpeed'
    # filterValue = 1770

    ##########################################################################
    ###########################  RETRIEVING DATA   ###########################
    ##########################################################################

    # Get all signals from desired sensor(s) and component(s)
    cols = fm.get_signal_list(SENSORS, COMPONENTS)

    # Starting path based on desired sensors or time periods
    startpath = fm.get_startpath(network_dir, SENSORS, YEAR, MONTH, DAY)

    CREATE_DATA_FILE = False # if False, data will be pickle-loaded from file
    LOAD_FILENAME = 'store_complete_data_set' # name of file to be loaded if not createDataFile

    # Get a dataframe containing desired data in desired formats
    if CREATE_DATA_FILE:
        data = fm.get_data(
                            root_dir=startpath,
                            cols=cols,
                            index_col=INDEX_COL,
                            chunksize=CHUNKSIZE,
                            filter_operation=FILTER_OPERATION
                        )
        if data.empty:
            print('No data in the selected interval qualifies the filtering conditions. No object has been pickled to memory.')
        else:
            store_filename = 'store'
            mem.store(data, store_filename)
            mem.store_time_interval(data.index[0], data.index[-1], store_filename + '_timeint')

    if not CREATE_DATA_FILE:
        data = mem.load(LOAD_FILENAME)
        timeint = mem.load_meta(LOAD_FILENAME + '_timeint')
        print(f'Data from {timeint[0]} to {timeint[-1]} loaded into memory.')

    ##########################################################################
    ###########################   CREATING MODEL  ############################
    ##########################################################################

    # Model parameters
    TRAINING_PCT=0.8
    TIMESTEPS=60


    CREATE_DATA_FILE = None # ! Not implemented functionality for this
    DO_TRANSFORM=True
    NORMAL_DIST = False # True if the data is known to have a normal distribution (changes transform function)
    DO_RESHAPE=True
    DELETE_PICKLED_FILES = None  # ! Not implemented functionality for this

    model = model.create(data,
                            prediction_cols=PREDICTION_COLS,
                            training_pct=TRAINING_PCT,
                            timesteps=TIMESTEPS,
                            create_data_file=CREATE_DATA_FILE,
                            do_transform=DO_TRANSFORM,
                            normal_dist=NORMAL_DIST,
                            do_reshape=DO_RESHAPE,
                            delete_pickled_files=DELETE_PICKLED_FILES
                        )

    # model.predict(...)


    ##########################################################################
    ########################### VISUALIZE RESULTS ############################
    ##########################################################################