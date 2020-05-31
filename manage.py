import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error


from src.funcs import file_management as filemag
from src.funcs import memory as mem
from src.modeling import helper_funcs as fnc
from src.modeling import lstm_sample as sample_model

if __name__ == '__main__':
    # Check if the user is connected to network drive containing data

    if sys.platform == 'win32':
        network_dir = filemag.check_access()
    else:
        raise NotImplementedError(
                                f"Operating system {sys.platform} is not " \
                                "supported in the current implementation."
                )
        sys.exit()

    ##########################################################################
    ###########################     INPUT DATA     ###########################
    ##########################################################################
    # Desired sensor(s) and component(s)
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
    FILE_SUFFIX = 'nov_2019'
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
    YEAR = 2019  # None: all available data will be used
    MONTH = 11 # None: all available data in given year will be used
    DAY = None # None: all available data in given month will be used

    # Reading and filtering of data
    INDEX_COL = 'time' # index column
    CHUNKSIZE = None # None: the model will train load all data simultaneously

    NORMAL_DIST = False # True if data has a normal distribution (affects transform function)
    DO_TRANSFORM = False
    DO_RESHAPE = False
    DO_MODELING = False
    DO_TESTING = False
    DELETE_PICKLED_FILES = None  # ! Not implemented functionality for this

    ##########################################################################
    ###########################  RETRIEVING DATA   ###########################
    ##########################################################################

    # Get all signals from desired sensor(s) and component(s)
    cols = filemag.get_signal_list(SENSORS, COMPONENTS)

    # Starting path based on desired sensors or time periods
    startpath = filemag.get_startpath(network_dir, SENSORS, YEAR, MONTH, DAY)

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
    else:
        data = mem.load(file_suffix=FILE_SUFFIX)
        if data.index.dtype == 'datetime64[ns]':
            tperiod = [data.index[0], data.index[-1]]
            print(
                f'Data from {tperiod[0]} to {tperiod[-1]} loaded into memory.'
            )

    ##########################################################################
    ###########################   CREATING MODEL  ############################
    ##########################################################################

    if PREDICTION_COLS is None:
        PREDICTION_COLS = data.columns

    # Data parameters
    TRAINING_PCT=1.0
    TIMESTEPS=30
    SCALER_TYPE = 'minmax'

    # Model parameters
    UNITS = 64
    RETURN_SEQUENCES = True
    DROPOUT_RATE = 0.2

    # Training parameters
    EPOCHS = 3
    BATCH_SIZE = 128

    if CREATE_DATA_FILE:
        NotImplementedError('Under development.')

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
    else:
        [scaler, df_train, df_test] = mem.load(
                                                file_prefix='transformed',
                                                file_suffix=FILE_SUFFIX
                                            )

    if DO_RESHAPE:
        [X_train, y_train, X_test, y_test] = fnc.get_reshaped(
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
    else:
        # ! REMOVE this, it is temporary to work with specific file names
        LOAD_TS = ''
        if TIMESTEPS in [5,10,15,20,25,30]:
            LOAD_TS = f'_ts-{str(TIMESTEPS).zfill(2)}'
        RESHAPED_SUFFIX = FILE_SUFFIX + LOAD_TS
        # ! REMOVE above
        [X_train, y_train, X_test, y_test] = mem.load(
                                                    file_prefix='reshaped',
                                                    file_suffix=RESHAPED_SUFFIX
                                                )

    if DO_MODELING:
        model = sample_model.create(
                                    X_train,
                                    y_train,
                                    UNITS=UNITS
                                )

        [model, history] = sample_model.train(
                                                model,
                                                X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                EPOCHS=EPOCHS,
                                                BATCH_SIZE=BATCH_SIZE
                                            )

        modelstring = fnc.get_modelstring(
                                            ep=EPOCHS,
                                            ts=TIMESTEPS,
                                            un=UNITS,
                                            bs=BATCH_SIZE
                                        )

        mem.save_model(
                        model,
                        history,
                        file_prefix='model',
                        modelstring=modelstring
                    )
    else:
        model, history = mem.load_from_list_of_models()
        # model, history = mem.load_model()

    if DO_TESTING:
        raise sys.exit('Under development.')
        # performance = sample_model.test_model(model,history)

    GET_FAULTY_DATA = False
    FAULTY_SUFFIX = 'faulty_data'

    if GET_FAULTY_DATA:
        faulty_data = filemag.get_and_store_data(
                            root_dir=startpath,
                            cols=cols,
                            index_col=INDEX_COL,
                            chunksize=CHUNKSIZE,
                            filter_operation=FILTER_OPERATION,
                            file_suffix=FAULTY_SUFFIX
                        )
        mem.store(faulty_data,file_suffix=FAULTY_SUFFIX)
    else:
        faulty_data = mem.load(file_suffix=FAULTY_SUFFIX)

    # faulty_data_transformed
    # X_test = None
    # y_test = None
    # pd = None
    # faulty_arr = np.array(faulty_data)
    # faulty_arr = faulty_arr.reshape(faulty_arr.shape[0], 1, faulty_arr.shape[1])
    # y_hat = model.predict(faulty_arr)

    # y_hat = model.predict(X_test)
    # y_faulty_hat = model.predict(X_faulty) # this is what we want



    # Need to inverse transform data
    y_hat = model.predict(X_test)
    if False: # old, transformed visualization
        y_pred1 = [row[0] for row in y_hat]
        y_pred2 = [row[1] for row in y_hat]
        y_real1 = [row[0] for row in y_test]
        y_real2 = [row[1] for row in y_test]
        df = pd.DataFrame()
        df['t1 (pred)'] = y_pred1
        df['t2 (pred)'] = y_pred2
        df['t1 (real)'] = y_real1
        df['t2 (real)'] = y_real2
        dft1 = df.filter(['t1 (pred)', 't1 (real)'])
        dft2 = df.filter(['t2 (pred)', 't2 (real)'])
        plt.plot(history['loss'], label = 'train')
        plt.plot(history['val_loss'], label = 'test')
        plt.legend()
        dft1.plot()
        plt.show()
        dft2.plot()
        plt.show()

    # Remove columns of predicted values and drop the first timestep
    df_hat = fnc.get_df_pred(
                                df_test,
                                y_hat,
                                TIMESTEPS,
                                prediction_cols=PREDICTION_COLS
                            )
    df_hat_inv = fnc.inverse_transform_dataframe(df_hat, scaler)
    df_test_inv = fnc.inverse_transform_dataframe(df_test, scaler)

    df_hat_filtered = df_hat_inv.filter(PREDICTION_COLS)
    df_test_filtered = df_test_inv[TIMESTEPS:].filter(PREDICTION_COLS)

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
    sns.distplot(mae_loss, kde=True)
    plt.show()
    THRESHOLD_PERCENTILE = 99
    threshold = np.percentile(mae_loss, THRESHOLD_PERCENTILE)

    below_threshold = mae_loss[mae_loss < threshold]
    above_threshold = mae_loss[mae_loss > threshold]

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
    for col in PREDICTION_COLS:
        df_hat_filtered[col] - df_test_filtered[col]

    performance['loss'] = mae_loss
    performance['threshold'] = threshold


    # model.predict(...)


    ##########################################################################
    ########################### VISUALIZE RESULTS ############################
    ##########################################################################


    # Stuff to do
    # - Make sure that the different functions work for training_pct = 1.0
    # - Check how to model.predict with faulty data - does it have to be transformed and/or reshaped?
    # - Add store from temp_file to manage for reshaped data
