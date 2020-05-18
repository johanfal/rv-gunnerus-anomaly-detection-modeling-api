import pickle

import numpy as np
import pandas as pd

from src.funcs import memory as mem
from src.models import helper_funcs as fnc
from tensorflow import keras

def create(data,training_pct=0.8,timesteps=100):
    CREATE_DATA_FILE = False
    DO_TRANSFORM = False
    DO_RESHAPE = False
    DELETE_PICKLED_FILES = False
    SIZE_LIMIT_TRAIN = 80000 # ! REMOVE, and remove dependies in DO_RESHAPE-logic
    SIZE_LIMIT_TEST = 20000 # ! REMOVE, and remove dependies in DO_RESHAPE-logic

    if DO_TRANSFORM:
        df_train, df_test = fnc.transform(data, training_pct)
        mem.store([df_train, df_test], 'transformed')
    else:
        [df_train, df_test] = mem.load('transformed')
    if DO_RESHAPE:
        X_train, y_train = fnc.reshape_data(df_train.head(SIZE_LIMIT_TRAIN), timesteps, bar_desc='Reshaping training data..')
        x_test, y_test = fnc.reshape_data(df_test.head(SIZE_LIMIT_TEST), timesteps, bar_desc='Reshaping test data..')
        reshaped = mem.store([X_train, y_train, x_test, y_test], 'reshaped')
    else:
        [X_train, y_train, x_test, y_test] = mem.load('reshaped')

    UNITS = 64
    RETURN_SEQUENCE = True
    RATE = 0.2

    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=UNITS,
                                input_shape=(X_train.shape[1], X_train.shape[2])
                                ))

    model.add(keras.layers.Dropout(rate=RATE))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=UNITS, return_sequences=RETURN_SEQUENCE))
    model.add(keras.layers.Dropout(rate=RATE))
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=X_train.shape[2])
        )
    )

    EPOCHS = 3
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.1
    SHUFFLE = False

    # The likely problem is that there is no chosen target. The output should be restricted to one
    # of the signal inputs. This is not the case as of now.
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=SHUFFLE
    )

    print('hello world.')


if __name__ == '__main__':
    import sys
    sys.exit('Run from manage.py, not model.')
