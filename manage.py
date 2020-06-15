#-----------------------------------------------------------------------------
# File: manage.py
# Purpose:
#   
#
# Created by: Johan Fredrik Alvsaker
# Last modified: 
#-----------------------------------------------------------------------------
# Standard library:
import sys

# External modules:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Local API:
from src.api import file_management as filemag
from src.api import memory as mem
from src.api import modeling_funcs as mfnc
from src.modeling import model_sample_lstm as sample_model

# Module import of your model, initially located in src/modeling/model.py:
from src.modeling import model as model # modify this if you change filename

#-----------------------------------------------------------------------------

# Check if the user is connected to network drive containing data
# network_dir = filemag.check_access() # supported for Windows OS
network_dir = 'Z'
# User inputs ----------------------------------------------------------------
# Sensor(s) and component(s):
SENSORS = 'NogvaEngine'
COMPONENTS = 'ME1' # optional

# Signals to predict (if None, all signals found from above are predicted):
PREDICTION_COLS = [
        'ME1_ExhaustTemp1',
        'ME1_ExhaustTemp2'
    ]

# File management ------------------------------------------------------------
CREATE_DATA_FILE = False # if False, data will be pickle-loaded from file
FILTER_OPERATION = True  # If True, only in-operation data will be used
FILE_SUFFIX = '2019'
REMOVE_FAULTY_DATA = True
DELETE_STORED_FILES = None  # ! Not implemented functionality for this

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
TRAINING_PERIOD =[
    None, # year (None: all available data will be used)
    None, # month (None: all available data in given year will be used)
    None # day (None: all available data in given month will be used)
    ]

# Reading and filtering of data
INDEX_COL = 'time' # index column
CHUNKSIZE = None # None: the model will load all data simultaneously

DO_TRANSFORM = False
DO_RESHAPE = False
DO_MODELING = False
DO_TESTING = True
GET_FAULTY = True
VISUALIZE_RESULTS = True

# Get all signals from desired sensor(s) and component(s)
signals = filemag.get_signal_list(SENSORS, COMPONENTS)
cols = list(signals.keys())

# Starting path based on desired sensors or time periods
startpath = filemag.get_startpath(network_dir, SENSORS, TRAINING_PERIOD)

# Data parameters
TRAINING_PCT=0.8
TIMESTEPS=30
SCALER_TYPE = 'minmax' # currently supported scalers: 'minmax', 'standard'
# (The scaler type affects data transformation. To add custom scalers, alter
# the get_scaler() function in src/modeling/helper_funcs.py as indicated.)

# Model parameters
UNITS = 32
RETURN_SEQUENCES = True
DROPOUT_RATE = 0.2

# Training parameters
EPOCHS = 20
BATCH_SIZE = 60

# Testing parameters
THRESHOLD_PCT = 97.25 # percentage of data not deemed anomalies
ANOMALY_NEIGHBORHOOD = 0   # necessary number of consecutive values exceeding
                            # a threshold to trigger an anomaly

# Get a dataframe containing desired data in desired formats
if CREATE_DATA_FILE:
    data = filemag.get_and_store_data(
                        root_dir=startpath,
                        cols=cols,
                        index_col=INDEX_COL,
                        chunksize=CHUNKSIZE,
                        filter_operation=FILTER_OPERATION,
                        file_suffix=FILE_SUFFIX,
                        faulty_data=FAULTY_DATA
                    )
else: # load stored data file
    data = mem.load(file_suffix=FILE_SUFFIX)
    if data.index.dtype == 'datetime64[ns]':
        tperiod = [data.index[0], data.index[-1]]
        print(
            f'Data from {tperiod[0]} to {tperiod[-1]} loaded into memory.'
        )

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
else: # load stored, transformed data
    [scaler, df_train, df_test] = mem.load(
                                            file_prefix='transformed',
                                            file_suffix=FILE_SUFFIX
                                        )

if DO_RESHAPE:
    [X_train, y_train, X_test, y_test] = mfnc.reshape(
                                            df_train,
                                            df_test,
                                            output_cols=PREDICTION_COLS,
                                            timesteps=TIMESTEPS
                                        )
    mem.store(
                [X_train, y_train, X_test, y_test],
                file_prefix='reshaped',
                file_suffix=FILE_SUFFIX
            )

else: # load stored, reshaped data
    [X_train, y_train, X_test, y_test] = mem.load(
                                                file_prefix='reshaped',
                                                file_suffix=FILE_SUFFIX
                                            )
    TIMESTEPS = X_train.shape[1] # redefine based on actual reshaped data

    if DO_MODELING:
        # Create model
        model = sample_model.create(
                                    X_train.shape[1:],
                                    UNITS=UNITS
                                )

        # Train model
        [model, history] = sample_model.train(
                                                model,
                                                X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                EPOCHS=EPOCHS,
                                                BATCH_SIZE=BATCH_SIZE
                                            )

        # Retrieve unique filename
        modelstring = mfnc.get_modelstring(
                                            ep=EPOCHS,
                                            ts=TIMESTEPS,
                                            un=UNITS,
                                            bs=BATCH_SIZE
                                        )

        # Save model and corresponding performance history
        mem.save_model(
                        model,
                        history,
                        file_prefix='model',
                        modelstring=modelstring
                    )
    else: # load stored model
        model, history = mem.load_from_list_of_models()

if GET_FAULTY:
    F_SUFFIX = 'faulty_data'
    ACTION_PARAMETERS = [
        False, # Create faulty data file
        False, # Tranform data
        False, # Reshape data
    ]
    # Choose time interval of data selection (remember that the interval with
    # simulated error must be included):
    F_INTERVAL =[
        2019, # year (None: all available data will be used)
        11, # month (None: all available data in given year will be used)
        None # day (None: all available data in given month will be used)
    ]

    f_startpath = filemag.get_startpath(network_dir,SENSORS,F_INTERVAL)

    [df_faulty, X_faulty, scaler_faulty]  = mfnc.get_faulty(
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


USE_TESTING_DATA = False # test and visualize with testing data
USE_FAULTY_DATA = True # test and visualize with faulty data

if DO_TESTING:
    # Predict values (use either testing or faulty data for this purpose):
    if USE_TESTING_DATA:
        [performance,absolute_error,thresholds] = sample_model.test(
                                    model,
                                    history,
                                    df_test=df_test,
                                    X_test=X_test,
                                    threshold_pct=THRESHOLD_PCT,
                                    anomaly_neighborhood=ANOMALY_NEIGHBORHOOD,
                                    pred_scaler=scaler,
                                    prediction_cols=PREDICTION_COLS
                                )

    if USE_FAULTY_DATA:
        [f_performance,f_absolute_error,f_thresholds] = sample_model.test(
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

if VISUALIZE_RESULTS:
    # Create plots (use either testing or faulty data for this purpose):
    if USE_TESTING_DATA:
        sample_model.visualize(
                                performance=performance,
                                history=history,
                                thresholds=thresholds,
                                absolute_error=absolute_error,
                                units=signals,
                            )
    if USE_FAULTY_DATA:
        sample_model.visualize(
                                performance=f_performance,
                                history=history,
                                thresholds=f_thresholds,
                                absolute_error=f_absolute_error,
                                units=signals,
                            )
