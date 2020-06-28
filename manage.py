# ----------------------------------------------------------------------------
# File: manage.py
# Purpose: the modeling API is run from the manage.py file, meaning that the
# actual 'model.py' file is called here. In addition to being the execution
# file, 'manage.py' also handles administrative variables, mostly related to
# file and memory management. By default, the file uses the user-made model
# file found in 'modeling/model.py'. If it is desired to use the example model
# instead, replace all calls to the model module with the example_model
# module.
#
# Created by: Johan Fredrik Alvsaker
# Last modified: 28.6.2020
# ----------------------------------------------------------------------------
# Standard library:
import sys

# Local API:
from src.api import file_management as filemag
from src.api import memory as mem
from src.api import modeling_funcs as mfnc
from src.modeling import model_example_lstm as example_model

# Module import of your model, initially located in src/modeling/model.py:
from src.modeling import model as model  # modify this if you change filename

# Network drive access -------------------------------------------------------
# Check if the user is connected to network drive containing data
network_dir = filemag.check_access()  # supported for Windows OS
# (if another operating systen is used, alter the network directory to
# coincide with the location of the network drive.)

# ------------------------------------------------------- Network drive access
# Sensor(s) and component(s) -------------------------------------------------
# The systems and components are defined in the 'rvgunnerus_systems.json'
SYSTEMS = 'NogvaEngine' # name taken from network drive folders
COMPONENTS = 'ME1'  # optional, set to None to select all system components

# Signals to predict:
PREDICTION_COLS = [
    'ME1_ExhaustTemp1',
    'ME1_ExhaustTemp2'
]
# Set PREDICTION_COLS to None to predict all signals from the components
# defined above (Warning! This might be load intensive)

# Name of index column:
INDEX_COL = 'time'
# (It is highly adviced using time as index column for visual purposes. The
# variable is defined here to change it if the time differs from the default.)

# ------------------------------------------------- Sensor(s) and component(s)
# File management ------------------------------------------------------------
# Declare the file suffix used throughout the execution:
FILE_SUFFIX = None
# (If new files are created during execution, these files will receive the
# designated file suffix from above. If files are loaded during execution,
# the program will look for files with the file suffix from above.)

CREATE_DATA_FILE = True  # if False, data will be pickle-loaded from file
FILTER_OPERATION = True  # If True, only in-operation data will be used
REMOVE_FAULTY_DATA = True  # If False, faulty data is not removed from data
# (It is important to exclude faulty data from the training data. Strictly
# prevent setting REMOVE_FAULTY_DATA to false if the data will be  used for
# model development.)

# Faulty data that will be removed if REMOVE_FAULTY_DATA is True:
FAULTY_DATA_INTERVAL = [
    [
        'NogvaEngine_20191121_105000.csv.gz',  # filename
        '2019-11-21 10:50:16.000',  # start of simulation error
        '2019-11-21 10:56:33.000'  # end of simulation error
    ]
]
# Must have three entries per faulty time interval:
#   - name of file containing error
#   - starting time of simulated error of specified time format
#   - ending time of simulated error of specified time format
# Time format: '%Y-%m-%d %H:%M:%S.%f'. Example: 2020-31-01 14:12:58.000'

# Desired time-period of training set:
TRAINING_PERIOD = [
    2019,  # year (None: all available data will be used)
    None,  # month (None: all available data in given year will be used)
    None  # day (None: all available data in given month will be used)
]

# Reading and filtering of data:
CHUNKSIZE = None  # None: the model will load all data simultaneously

# ------------------------------------------------------------ File management
# Modeling operations --------------------------------------------------------
DO_TRANSFORM = True  # create scaler and scaled training and testing data
DO_RESHAPE = True  # reshape data based on provided timesteps
DO_MODELING = True  # create and train new model
DO_TESTING = True  # test model on training data
VISUALIZE_RESULTS = True  # Create plots visualizing modeling results

# Create scaled faulty data, reshaped faulty data, and faulty scaler:
GET_FAULTY = True
# (Although there is previous REMOVE_FAULTY_DATA_INTERVAL option, the
# GET_FAULTY data allows for testing and visualizing a model on faulty data in
# the same execution as modeling is performed. There are additional parameters
# that must be considered, be sure to check these variables in the
# 'Get faulty' section below.)

USE_TESTING_DATA = False  # test and visualize with testing data
USE_FAULTY_DATA = not USE_TESTING_DATA  # test and visualize with faulty data
# (Decide what data to use for testing and visualization of results. Either
# USE_TESTING_DATA or USE_FAULTY_DATA must be True, but both cannot be True at
# the same time. Furthermore, if USE_FAULTY_DATA is True GET_FAULTY must also
# be True.)

# ---------------------------------------------------------Modeling operations
# Data, modeling and training parameters--------------------------------------

# Below are a set of parameters used for data management, modeling and
# training. Many of these parameters may be useful to define in their
# respective use-files, e.g. the model parameters can be defined where the
# modeling is performed. If so, remember to remove the model parameters from
# the functions called in this 'manage.py' file. The easiest way of finding
# these is to CTRL+F the model parameters defined above.)

# Data parameters:
TRAINING_PCT = 0.8  # fraction of data used for training sets
# (the remainder will be used for testing)
TIMESTEPS = 30  # timesteps used per prediction
# (If timesteps is e.g. 10, each output value is predicted based on the last
# 10 timesteps at the point of prediciton.)
SCALER_TYPE = 'minmax'  # currently supported scalers: 'minmax', 'standard'
# (The scaler type affects data transformation. To add custom scalers, alter
# the get_scaler() function in src/modeling/helper_funcs.py as indicated.)

# Model parameters:
UNITS_LSTM = 32
UNITS_DENSE = 2
DROPOUT_RATE = 0.2

# Training parameters:
EPOCHS = 60  # number of training repetition cycles
# (One cycle is complete when the model has gone through one set of training
# data samples.)
BATCH_SIZE = 600  # samples processed before model is updated
# (larger batch size will decrease the update granularity, and consequently
# decrease runtime. Therefore, there is a trade-off in the batch size choice.)

# Testing parameters:
THRESHOLD_PCT = 95  # percentage of data not deemed anomalies
# (The THRESHOLD_PCT can be either a scalar value or a list of values in
# appropriate order according to the desired output columns to be predicted.
# If a list of values is input, the threshold values will be calculated with
# the threshold percentage for the specific prediction column.)
ANOMALY_NEIGHBORHOOD = 5  # necessary number of consecutive values exceeding
# a threshold to trigger an anomaly
# (THRESHOLD_PCT and ANOMALY_NEIGHBORHOOD should be used to tune the results
# in order to retrieve a useful threshold value. By filtering out outliers the
# threshold value decreases to a more probable value.)

# ------------------------------------- Data, modeling and training parameters
# Filtering, processing, transforming and reshaping data ---------------------

# Get all signals from desired system(s) and component(s):
systems = filemag.get_system_list(SYSTEMS, COMPONENTS)
cols = list(systems.keys())

# Starting path based on desired sensors or time periods:
startpath = filemag.get_startpath(network_dir, SYSTEMS, TRAINING_PERIOD)

# Get dataframe containing desired data in desired formats:
if CREATE_DATA_FILE:
    data = filemag.get_and_store_data(
        root_dir=startpath,
        cols=cols,
        index_col=INDEX_COL,
        chunksize=CHUNKSIZE,
        filter_operation=FILTER_OPERATION,
        file_suffix=FILE_SUFFIX,
        faulty_data=FAULTY_DATA_INTERVAL
    )
else:  # load stored data file:
    data = mem.load(file_suffix=FILE_SUFFIX)
    if data.index.dtype == 'datetime64[ns]':
        tperiod = [data.index[0], data.index[-1]]
        print(
            f'Data from {tperiod[0]} to {tperiod[-1]} loaded into memory.'
        )

# Get scaled training and testing data, together with scaler:
if DO_TRANSFORM:
    [scaler, df_train, df_test] = mfnc.transform(
        data,
        TRAINING_PCT,
        scaler_type=SCALER_TYPE
    )
    mem.store(
        [scaler, df_train, df_test],
        file_prefix='transformed',
        file_suffix=FILE_SUFFIX
    )
else:  # load stored, transformed data:
    [scaler, df_train, df_test] = mem.load(
        file_prefix='transformed',
        file_suffix=FILE_SUFFIX
    )

# Get reshaped training and testing data as X and y:
# (X has shape (timesteps, features), while y has shape (1, features).)
if DO_RESHAPE:
    [X_train, y_train, X_test, y_test] = mfnc.reshape(
        df_train,
        df_test,
        output_cols=PREDICTION_COLS,
        timesteps=TIMESTEPS
    )
    # Store reshaped data:
    mem.store(
        [X_train, y_train, X_test, y_test],
        file_prefix='reshaped',
        file_suffix=FILE_SUFFIX
    )

else:  # load stored, reshaped data:
    [X_train, y_train, X_test, y_test] = mem.load(
        file_prefix='reshaped',
        file_suffix=FILE_SUFFIX
    )
    # Redefine if the given value does not coincide with the provided file:
    TIMESTEPS = X_train.shape[1]

# --------------------- Filtering, processing, transforming and reshaping data
# Modeling and training ------------------------------------------------------

# Create and test model and save resulting model and history files:
if DO_MODELING:
    # Create model:
    model = model.create(
        X_train.shape[1:],
        UNITS_LSTM=UNITS_LSTM,
        UNITS_DENSE=UNITS_DENSE,
        DROPOUT_RATE=DROPOUT_RATE,
    )

    # Train model:
    [model, history] = model.train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        EPOCHS=EPOCHS,
        BATCH_SIZE=BATCH_SIZE
    )

    # Retrieve unique filename:
    modelstring = mfnc.get_modelstring(
        ep=EPOCHS,
        ts=TIMESTEPS,
        un=UNITS_LSTM,
        bs=BATCH_SIZE
    )

    # Save model and corresponding performance history:
    mem.save_model(
        model,
        history,
        file_prefix='model',
        modelstring=modelstring
    )
else:  # load stored model
    model, history = mem.load_from_list_of_models()

# ------------------------------------------------------ Modeling and training
# Get faulty data ------------------------------------------------------------

if GET_FAULTY:
    F_SUFFIX = 'faulty_data'
    ACTION_PARAMETERS = [
        True,  # Create faulty data file
        True,  # Tranform data
        True,  # Reshape data
    ]
    # Choose time interval of data selection (remember that the interval with
    # simulated error, defined in FAULTY_DATA_INTERVAL, must be included):
    F_INTERVAL = [
        2019,  # year (None: all available data will be used)
        11,  # month (None: all available data in given year will be used)
        21  # day (None: all available data in given month will be used)
    ]

    # Get starting path for faulty data:
    f_startpath = filemag.get_startpath(network_dir, SYSTEMS, F_INTERVAL)

    # Get transformed data, reshaped data, and scaler:
    [df_faulty, X_faulty, scaler_faulty] = mfnc.get_faulty(
        root_dir=f_startpath,
        cols=cols,
        timesteps=TIMESTEPS,
        index_col=INDEX_COL,
        chunksize=CHUNKSIZE,
        filter_operation=FILTER_OPERATION,
        faulty_suffix=F_SUFFIX,
        action_parameters=ACTION_PARAMETERS,
        scaler_type=SCALER_TYPE,
        output_cols=PREDICTION_COLS
    )

# ------------------------------------------------------------ Get faulty data
# Perform testing ------------------------------------------------------------

if DO_TESTING:
    # Predict values using testing data:
    if USE_TESTING_DATA:
        [performance, absolute_error, thresholds] = model.test(
            model,
            history,
            df_test=df_test,
            X_test=X_test,
            threshold_pct=THRESHOLD_PCT,
            anomaly_neighborhood=ANOMALY_NEIGHBORHOOD,
            pred_scaler=scaler,
            prediction_cols=PREDICTION_COLS
        )

    # Predict values using faulty data:
    if USE_FAULTY_DATA:
        [f_performance, f_absolute_error, f_thresholds] = model.test(
            model,
            history,
            df_test=df_faulty,
            X_test=X_faulty,
            threshold_pct=THRESHOLD_PCT,
            anomaly_neighborhood=ANOMALY_NEIGHBORHOOD,
            pred_scaler=scaler,
            test_scaler=scaler_faulty,
            prediction_cols=PREDICTION_COLS
        )

# ------------------------------------------------------------ Perform testing
# Visualize results ----------------------------------------------------------

if VISUALIZE_RESULTS:
    # Create plots using testing data:
    if USE_TESTING_DATA:
        model.visualize(
            performance=performance,
            history=history,
            thresholds=thresholds,
            absolute_error=absolute_error,
            units=systems,
        )

    # Create plots using faulty data:
    if USE_FAULTY_DATA:
        model.visualize(
            performance=f_performance,
            history=history,
            thresholds=f_thresholds,
            absolute_error=f_absolute_error,
            units=systems,
        )
