import pickle

import numpy as np
import pandas as pd

from src.funcs import memory as mem
from src.models import helper_funcs as fnc
from tensorflow import keras

def create(data,training_pct=0.8,timesteps=100):
    CREATE_DATA_FILE = False # ! Not implemented functionality for this
    DO_TRANSFORM = True
    NORMAL_DIST = False # True if the data is known to have a normal distribution (changes transform function)
    DO_RESHAPE = True
    DELETE_PICKLED_FILES = False  # ! Not implemented functionality for this
    SIZE_LIMIT_TRAIN = 80000 # ! REMOVE, and remove dependies in DO_RESHAPE-logic
    SIZE_LIMIT_TEST = 20000 # ! REMOVE, and remove dependies in DO_RESHAPE-logic
    OUTPUT_COLS = ['ME1_ExhaustTemp1','ME1_ExhaustTemp2'] # desired columns to predict the values of. If None, all values will be predicted

    if DO_TRANSFORM:
        scaler, df_train, df_test = fnc.transform(data, training_pct, normal_dist=NORMAL_DIST)
        mem.store([scaler, df_train, df_test], 'transformed')
    else:
        [scaler, df_train, df_test] = mem.load('transformed')
    if DO_RESHAPE:
        X_train, y_train = fnc.reshape_data(df_train,timesteps,output_cols=OUTPUT_COLS,bar_desc='Reshaping training data..')
        X_test, y_test = fnc.reshape_data(df_test,timesteps,output_cols=OUTPUT_COLS,bar_desc='Reshaping test data..')
        print("Storing reshaped dataframes as 'src/datastore/reshaped.pckl'.")
        reshaped = mem.store([X_train, y_train, X_test, y_test], 'reshaped')
        print("Data succesfully reshaped and stored.")
        print(f"Dimensions: X_train({X_train.shape}) | y_train({y_train.shape}) | X_test({X_test.shape}) | y_test ({y_test.shape})")
    else:
        [X_train, y_train, X_test, y_test] = mem.load('reshaped')

    UNITS = 32
    RETURN_SEQUENCE = True
    RATE = 0.2

    model = keras.Sequential()

    # OLD MODEL
    # model.add(keras.layers.LSTM(units=UNITS,
    #                             input_shape=(X_train.shape[1], X_train.shape[2])
    #                             ))

    # model.add(keras.layers.Dropout(rate=RATE))
    # model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    # model.add(keras.layers.LSTM(units=UNITS, return_sequences=RETURN_SEQUENCE))
    # model.add(keras.layers.Dropout(rate=RATE))
    # model.add(
    #     keras.layers.TimeDistributed(
    #         keras.layers.Dense(units=X_train.shape[2])
    #     ))

    # NEW MODEL
    model.add(keras.layers.LSTM(UNITS, input_shape=(X_train.shape[1:])))
    model.add(keras.layers.Dense(y_train.shape[1])) # 12 output variables

    EPOCHS = 15
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.1
    SHUFFLE = False

    model.compile(loss='mae', optimizer='adam')
    model.summary()

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=2, shuffle=False)

    y_hat = model.predict(X_test)

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

    import matplotlib.pyplot as plt
    dft1.plot()
    plt.show()
    dft2.plot()
    plt.show()

    # history = model.fit(
    #     X_train,
    #     y_train,
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     validation_split=VALIDATION_SPLIT,
    #     shuffle=SHUFFLE
    # )

    print('hello world.')
    return model


if __name__ == '__main__':
    import sys
    sys.exit('Run from manage.py, not model.')
