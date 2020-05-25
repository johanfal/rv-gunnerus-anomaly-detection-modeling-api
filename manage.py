import sys

# Custom module imports
from src.funcs import file_management as filemag
from src.funcs import memory as mem
from src.modeling import lstm_sample as sample_model
from src.modeling import helper_funcs as fnc

if __name__ == '__main__':
    # Check if the user is connected to network drive containing data

    model, history = mem.load_from_list_of_models()

    if sys.platform == 'win32':
        network_dir = filemag.check_access()
    else:
        raise NotImplementedError(
                    f"Support for {sys.platform} has yet to be implemented."
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
    CREATE_DATA_FILE = True # if False, data will be pickle-loaded from file
    FILTER_OPERATION = True  # If True, only in-operation data will be used
    FILE_SUFFIX = 'complete_data_set'
    REMOVE_FAULTY_DATA = True

    # Faulty data that will be removed if REMOVE_FAULTY_DATA is True
    # Must have three entries (name of file containing error, and starting and ending timestamps of simulated error)
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

    NORMAL_DIST = False # True if the data is known to have a normal distribution (changes transform function)
    CREATE_DATA_FILE = False # ! Not implemented functionality for this
    DO_TRANSFORM=False
    DO_RESHAPE=False
    DO_MODELING = True
    DO_TESTING = True
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
        data = filemag.get_data(
                            root_dir=startpath,
                            cols=cols,
                            index_col=INDEX_COL,
                            chunksize=CHUNKSIZE,
                            filter_operation=FILTER_OPERATION
                        )
        if data.empty:
            print('No data in the selected interval qualifies the filtering conditions. No object has been pickled to memory.')
        else:
            mem.store(data,file_suffix=FILE_SUFFIX)
            mem.store_time_interval(data.index[0], data.index[-1],file_suffix=FILE_SUFFIX)

    if not CREATE_DATA_FILE:
        data, timeint = mem.load(file_suffix=FILE_SUFFIX), mem.load_time_interval(file_suffix=FILE_SUFFIX)
        print(f'Data from {timeint[0]} to {timeint[-1]} loaded into memory.')

    ##########################################################################
    ###########################   CREATING MODEL  ############################
    ##########################################################################

    if PREDICTION_COLS is None:
        PREDICTION_COLS = data.columns

    # Model parameters
    TRAINING_PCT=0.8
    TIMESTEPS=15

    if CREATE_DATA_FILE:
        NotImplementedError('Under development.')

    if DO_TRANSFORM:
        [scaler, df_train, df_test] = fnc.transform(data, TRAINING_PCT, normal_dist=NORMAL_DIST)
        mem.store([scaler, df_train, df_test], FILE_SUFFIX)
    else:
        [scaler, df_train, df_test] = mem.load('transformed',FILE_SUFFIX)

    if DO_RESHAPE:
        [X_train, y_train, X_test, y_test] = fnc.get_reshaped(df_train,df_test,output_cols=PREDICTION_COLS,timesteps=TIMESTEPS)
    else:
        [X_train, y_train, X_test, y_test] = mem.load('reshaped',FILE_SUFFIX)


    # Model parameters
    UNITS = 64
    RETURN_SEQUENCES = True
    SCALE = 0.2

    # Training parameters
    EPOCHS = 10
    BATCH_SIZE = 128

    if DO_MODELING:
        model = sample_model.create(X_train,y_train,UNITS,RETURN_SEQUENCES,SCALE)
        [model, history] = sample_model.train(model,X_train,y_train,X_test,y_test,EPOCHS,BATCH_SIZE)
        modelstring = fnc.get_modelstring(ep=EPOCHS,
                                        ts=TIMESTEPS,
                                        un=UNITS,
                                        bs=BATCH_SIZE
                                    )
        mem.save_model(model,history,file_prefix='model',modelstring=modelstring)
    else:
        model, history = mem.load_from_list_of_models()
        # model, history = mem.load_model()

    if DO_TESTING:
        NotImplementedError('Under development.')
        # performance = sample_model.test_model(model,history)

    faulty_data = mem.load(file_suffix='faulty_data')
    X_test = None
    y_test = None
    pd = None
    y_hat = model.predict(X_test)

    # train_mae_loss = np.mean(np.abs(y_hat))

    # Need to inverse transform data
    y_pred1 = [row[0][0] for row in y_hat]
    y_pred2 = [row[0][1] for row in y_hat]
    y_real1 = [row[0] for row in y_test]
    y_real2 = [row[1] for row in y_test]

    df = pd.DataFrame()
    df['t1 (pred)'] = y_pred1
    df['t2 (pred)'] = y_pred2
    df['t1 (real)'] = y_real1
    df['t2 (real)'] = y_real2

    dft1 = df.filter(['t1 (pred)', 't1 (real)'])
    dft2 = df.filter(['t2 (pred)', 't2 (real)'])

    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'test')
    plt.legend()
    dft1.plot()
    plt.show()
    dft2.plot()
    plt.show()

    # model.predict(...)


    ##########################################################################
    ########################### VISUALIZE RESULTS ############################
    ##########################################################################