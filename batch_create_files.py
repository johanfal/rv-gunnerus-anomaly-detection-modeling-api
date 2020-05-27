import sys

# Custom module imports
from src.funcs import file_management as filemag
from src.funcs import memory as mem
from src.modeling import lstm_sample as sample_model
from src.modeling import helper_funcs as fnc


SENSORS = 'NogvaEngine'
COMPONENTS = 'ME1' # optional
PREDICTION_COLS = [ # desired columns to predict the values of. If None, all values will be predicted
        'ME1_ExhaustTemp1',
        'ME1_ExhaustTemp2'
    ]

# COMPONENTS = {'NogvaEngine':['ME1', 'ME2']
#             }

# File management
CREATE_DATA_FILE = False # if False, data will be pickle-loaded from file
FILTER_OPERATION = True  # If True, only in-operation data will be used
REMOVE_FAULTY_DATA = False

# Faulty data that will be removed if REMOVE_FAULTY_DATA is True
# Must have three entries per faulty time interval:
#   - name of file containing error
#   - starting time of simulated error of specified time format
#   - ending time of simulated error of specified time format
# Time format: '%Y-%m-%d %H:%M:%S.%f'. Example: 2020-31-01 14:12:58.000'
FAULTY_DATA = [
    [
        'NogvaEngine_20191121_105000.csv.gz', # filename
        '2019-11-21 10:50:16.000', # start of simulation error
        '2019-11-21 10:56:33.000' # end of simulation error
    ]
]

# Desired time-period of training set
# § IMPROVEMENT: should be able to select range of years, months, or days

INDEX_COL = 'time' # index column
CHUNKSIZE = None # None: the model will train load all data simultaneously

NORMAL_DIST = False # True if data has a normal distribution (affects transform function)
DO_TRANSFORM = False
DO_RESHAPE = False
DO_MODELING = False
DO_TESTING = False
DELETE_PICKLED_FILES = None  # ! Not implemented functionality for this
TRAINING_PCT = 0.8

cols = filemag.get_signal_list(SENSORS, COMPONENTS)
# Starting path based on desired sensors or time periods

new_files = {
    0:[
        'nov_2019',
        2019,
        11,
        None
    ],
    1:[
        '2019',
        2019,
        None,
        None
    ],
    2:[
        '2020',
        2020,
        None,
        None
    ],
    3:[
        'complete_data',
        None,
        None,
        None
    ],
    4:[
        'des_2019',
        2019,
        12,
        None
    ],
    5:[
        'jan_2020',
        2020,
        1,
        None
    ],
    6:[
        'feb_2020',
        2020,
        2,
        None
    ],
    7:[
        'mar_2020',
        2020,
        3,
        None
    ],
    8:[
        'apr_2020',
        2020,
        4,
        None
    ],
    9:[
        'may_2020',
        2020,
        5,
        None
    ]
}

FILE_SUFFIX = None
YEAR = 2019  # None: all available data will be used
MONTH = 11 # None: all available data in given year will be used
DAY = None # None: all available data in given month will be used

tss = [5, 10, 15, 20, 25, 30]
# tss = [45, 60, 90, 120]

for i in range(len(new_files)):
    FILE_SUFFIX = new_files[i][0]
    print('File: ',FILE_SUFFIX)
    YEAR = new_files[i][1]
    MONTH = new_files[i][2]
    DAY = new_files[i][3]
    startpath = filemag.get_startpath('Z:', SENSORS, YEAR, MONTH, DAY)
    # data = filemag.get_and_store_data(
    #                     root_dir=startpath,
    #                     cols=cols,
    #                     index_col=INDEX_COL,
    #                     chunksize=CHUNKSIZE,
    #                     filter_operation=FILTER_OPERATION,
    #                     file_suffix=FILE_SUFFIX,
    #                     faulty_data=FAULTY_DATA
    #                 )
    data = mem.load(file_suffix=FILE_SUFFIX)

    if type(data) == bool:
        if data == False:
            continue

    [scaler, df_train, df_test] = fnc.transform(
                                                    data,
                                                    TRAINING_PCT,
                                                    normal_dist=NORMAL_DIST
                                                )
    mem.store(
                [scaler, df_train, df_test],
                file_prefix='transformed',
                file_suffix=FILE_SUFFIX
            )
    for ts in tss:
        print('Reshaping data with timesteps: ',ts)
        [X_train, y_train, X_test, y_test] = fnc.reshape(
                                                    df_train,
                                                    df_test,
                                                    output_cols=PREDICTION_COLS,
                                                    timesteps=ts,
                                                    verbose=True
                                                )
        mem.store(
            [X_train, y_train, X_test, y_test],
            file_prefix='reshaped',
            file_suffix=FILE_SUFFIX + '_ts-' + str(ts).zfill(2)
        )