import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from src.funcs import file_management as filemag
from src.funcs import memory as mem
from src.modeling import helper_funcs as fnc
from src.modeling import model_sample_lstm as sample_model

# Module import of your model, initially located in src/modeling/model.py:
from src.modeling import model as model # modify this if you change filename

if True: # testing faulty
    from datetime import datetime
    result = mem.load(file_prefix='faulty_testing_results')
    for key,res in result.items():
        p = res[0]['ME1_ExhaustTemp1']
        tot_anoms = p.anom.sum()
        tot_vals = p.shape[0]
        p = p.loc[p.index[10500:12800]]
        start_marker = p.index.get_loc('2019-11-21 10:50:16')
        end_marker = p.index.get_loc('2019-11-21 10:56:33.000')
        start_marker = datetime.strptime(str(p.index[start_marker]), '%Y-%m-%d %H:%M:%S')
        end_marker = datetime.strptime(str(p.index[end_marker]), '%Y-%m-%d %H:%M:%S')
        plt.plot(p.index, p.loss, label='loss')
        plt.plot(p.index, p.loss.max()*p.anom, label='anom')
        plt.plot(p.index, [res[2]['ME1_ExhaustTemp1']]*p.shape[0], label=f"threshold: {res[2]['ME1_ExhaustTemp1']:.2f}")
        plt.axvspan(start_marker, end_marker, color='red', alpha=0.3)
        sum1 = p.anom.sum()
        vals = p.shape[0]
        plt.title(key + f" (anoms/vals: {sum1}/{vals})\n(tot anoms/vals: {tot_anoms}/{tot_vals})")
        plt.legend()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
    sys.exit()
# Check if the user is connected to network drive containing data
network_dir = filemag.check_access() # supported for Windows OS

##########################################################################
###########################     INPUT DATA     ###########################
##########################################################################

# Sensor(s) and component(s)
SENSORS = 'NogvaEngine'
COMPONENTS = 'ME1' # optional

# Signals to predict. If None, all values found will be predicted
PREDICTION_COLS = [
        'ME1_ExhaustTemp1',
        'ME1_ExhaustTemp2'
    ]

# File management
CREATE_DATA_FILE = False # if False, data will be pickle-loaded from file
FILTER_OPERATION = True  # If True, only in-operation data will be used
FILE_SUFFIX = 'nov_2019'
REMOVE_FAULTY_DATA = False
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
    2019, # year (None: all available data will be used)
    11, # month (None: all available data in given year will be used)
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

##########################################################################
###########################  RETRIEVING DATA   ###########################
##########################################################################

# Get all signals from desired sensor(s) and component(s)
cols = filemag.get_signal_list(SENSORS, COMPONENTS)

# Starting path based on desired sensors or time periods
startpath = filemag.get_startpath(network_dir, SENSORS, TRAINING_PERIOD)


##########################################################################
###########################   CREATING MODEL  ############################
##########################################################################

# Data parameters
TRAINING_PCT=0.8
TIMESTEPS=30
SCALER_TYPE = 'minmax' # currently supported scalers: 'minmax', 'standard'
# (The scaler type affects data transformation. To add custom scalers, alter
# the get_scaler() function in src/modeling/helper_funcs.py as indicated.)

# Model parameters
UNITS = 64
RETURN_SEQUENCES = True
DROPOUT_RATE = 0.2

# Training parameters
EPOCHS = 3
BATCH_SIZE = 128

# Testing parameters
THRESHOLD_PCT = 99.95 # percentage of data not deemed anomalies
ANOMALY_NEIGHBORHOOD = 17   # necessary number of consecutive values exceeding
# 4: (17,4), 6: (13,0), 10: (7,0), 15: (2,0), 16: (1,0), 17:(0,0) (T_PCT: 99.95)
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
    [scaler, df_train, df_test] = fnc.transform(
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
    [X_train, y_train, X_test, y_test] = fnc.reshape(
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
else: # laod stored, reshaped data
    # ! REMOVE this, it is temporary to work with specific filenames
    LOAD_TS = ''
    if TIMESTEPS in [5,10,15,20,25,30]:
        LOAD_TS = f'_ts-{str(TIMESTEPS).zfill(2)}'
    RESHAPED_SUFFIX = FILE_SUFFIX + LOAD_TS
    # ! REMOVE above
    [X_train, y_train, X_test, y_test] = mem.load(
                                                file_prefix='reshaped',
                                                file_suffix=RESHAPED_SUFFIX
                                            )
    TIMESTEPS = X_train.shape[1] # redefine based on actual reshaped data

    if DO_MODELING:
        # Create model
        model = sample_model.create(
                                    X_train.shape[1:],
                                    y_train,
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
        modelstring = fnc.get_modelstring(
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
        21 # day (None: all available data in given month will be used)
    ]

    f_startpath = filemag.get_startpath(network_dir,SENSORS,F_INTERVAL)

    [df_faulty, X_faulty, scaler_faulty]  = fnc.get_faulty(
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

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

if DO_TESTING:
    # Predict values (use either testing or faulty data for this purpose):
    # [performance,absolute_error,thresholds] = sample_model.test(
    #                                 model,
    #                                 history,
    #                                 df_test=df_test,
    #                                 X_test=X_test,
    #                                 threshold_pct=THRESHOLD_PCT,
    #                                 anomaly_neighborhood=ANOMALY_NEIGHBORHOOD,
    #                                 pred_scaler=scaler,
    #                                 prediction_cols=PREDICTION_COLS
    #                             )


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

VISUALIZE_RESULTS = True
if VISUALIZE_RESULTS:
    sample_model.visualize(
                            performance=performance,
                            history=history,
                            thresholds=thresholds,
                            absolute_error=absolute_error
                        )

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

y_hat = model.predict(X_test) # 
y_hat_test = model.predict(X_test) # 
y_hat_faulty = model.predict(X_faulty) # 

# Create a dataframe inserting predicted values into relevant columns:
df_hat = fnc.get_df_pred( # 
                            df_test, # 
                            y_hat, # 
                            prediction_cols=PREDICTION_COLS # 
                    ) # 
df_hat_inv = fnc.inverse_transform_dataframe(df_hat, scaler)
df_test_inv = fnc.inverse_transform_dataframe(df_test, scaler)

df_hat_filtered = df_hat_inv.filter(PREDICTION_COLS) # 
df_test_filtered = df_test_inv[TIMESTEPS:].filter(PREDICTION_COLS) # 
from sklearn.metrics import mean_squared_error
rmse_ind = []
mae_ind = []
for col in df_hat_filtered:
    rmse_ind.append(np.sqrt(
            mean_squared_error(df_hat_filtered[col],df_test_filtered[col])
        ))
    mae_ind.append(np.mean(np.abs(df_hat_filtered[col].values-df_test_filtered[col].values)))
rmse_tot = np.sqrt(mean_squared_error(df_hat_filtered,df_test_filtered))
mae_tot = np.mean(np.abs(df_hat_filtered-df_test_filtered))

df_hat_arr = np.array(df_hat_filtered)
df_test_arr = np.array(df_test_filtered)
hat_rs = df_hat_arr.reshape(df_hat_arr.shape[0], 1, df_hat_arr.shape[1])
test_rs = df_test_arr.reshape(df_test_arr.shape[0], 1, df_test_arr.shape[1])

mae_loss = np.mean(np.abs(hat_rs-test_rs), axis=1)
if False: # plotting
    sns.distplot(mae_loss, kde=True)
    plt.show()
THRESHOLD_PERCENTILE = 99.95
NEIGHBORS = 10

##############################################################################
##############################################################################
##############################################################################
##############################################################################


threshold = np.percentile(mae_loss, THRESHOLD_PERCENTILE)
below_threshold = mae_loss[mae_loss < threshold]
above_threshold = mae_loss[mae_loss > threshold]

if False: # plotting
    sns.distplot(below_threshold,kde=True,label=f'below, n={below_threshold.shape[0]} (max: {below_threshold.max():.2f})')
    sns.distplot(above_threshold,kde=True,label=f'above, n={above_threshold.shape[0]} (max: {above_threshold.max():.2f})')
    plt.legend()
    plt.show()
    sns.distplot(mae_loss[mae_loss < threshold], kde=True)
    plt.show()
    sns.distplot(mae_loss[mae_loss > np.percentile(mae_loss, 99.9)], kde=True)
    plt.show()
    plt.plot(history['loss'], label = 'train')
    plt.plot(history['val_loss'], label = 'test')
    plt.legend()
    plt.show()
    ax = df_hat_filtered['ME1_ExhaustTemp1'].plot()
    df_test_filtered['ME1_ExhaustTemp1'].plot(ax=ax)
    plt.legend()
    plt.show()
    ax = df_hat_filtered['ME1_ExhaustTemp2'].plot()
    df_test_filtered['ME1_ExhaustTemp2'].plot(ax=ax)
    plt.legend()
    plt.show()

performance = pd.DataFrame(index=df_hat.index)
performance['threshold'] = threshold # ! REMOVE, probably unnecessary
col_counter = 0
for col in PREDICTION_COLS:
    df_hat_filtered[col] - df_test_filtered[col]
    performance[f'loss_{col}'] = [row[col_counter] for row in mae_loss]
    performance[f'anomaly_{col}'] = performance[f'loss_{col}'] > performance.threshold
    performance[f'anomaly_{col}_neigh'] = fnc.get_anomaly_range(performance[f'loss_{col}'],threshold,neighbors=NEIGHBORS)
    performance[f'pred_{col}'] = df_hat_filtered[col]
    performance[f'real_{col}'] = df_test_filtered[col]
    all_anomalies = performance.index[performance[f'anomaly_{col}'] == True].tolist()
    true_anomalies = performance.index[performance[f'anomaly_{col}_neigh'] == True].tolist()
    false_anomalies = np.setdiff1d(all_anomalies, true_anomalies)
    for anom in false_anomalies:
        performance.drop(anom, inplace=True)


    plt.plot(performance.index, performance[f'loss_{col}'], label=f'loss_{col}')
    plt.plot(performance.index, performance[f'loss_{col}'].max()* performance[f'anomaly_{col}_neigh'], label=f'anomaly_{col}')
    plt.plot(performance.index, performance.threshold, label='threshold')
    plt.legend()
    plt.show()

    col_counter += 1

# Stuff to do
# - Check how to model.predict with faulty data - does it have to be transformed and/or reshaped?
# - Add store from 'batch_create_files.py' to 'manage.py' for reshaped data
