import pickle

import numpy as np
import pandas as pd

from src.funcs import memory as mem
from src.modeling import helper_funcs as fnc
from tensorflow import keras

def create(X_train,y_train,**parameters):
    """Description. The function takes a variable number of keyword arguments,
    which can be used to build the model."""

    model = keras.Sequential()

    # Create variables based on the desired keyword arguments used to build
    # the model. These must be changed in accordance with the **parameters.
    UNITS = parameters['UNITS']
    RETURN_SEQUENCE = parameters['RETURN_SEQUENCE']
    RATE = parameters['RATE']
    OPTIMIZER = 'adam'  # try out different optimizer (dynamic loss rate?)

    model.add(keras.layers.LSTM(UNITS, input_shape=(X_train.shape[1:])))
    model.add(keras.layers.Dropout(rate=RATE))
    model.add(keras.layers.Dense(2, input_dim = X_train.shape[1]))
    model.add(keras.layers.RepeatVector(n=1)) # This was the last thing you changed, and it seemed to be working..
    # ! Is it possible to replace the hard-coded value of 1 with the shape of y_train, somehow?
    model.add(keras.layers.LSTM(UNITS, return_sequences=True))
    model.add(keras.layers.Dropout(rate=RATE))
    # model.add(keras.layers.Dense(2))
    model.add(keras.layers.TimeDistributed(
    keras.layers.Dense(2)) # change this to use the shape of y_train
    )

    model.compile(loss='mae', optimizer=OPTIMIZER)

    model.summary() # optional printout of key model properties

    return model

def create_old_model(X_train,y_train,**parameters):
    """Description. The function takes a variable number of keyword arguments,
    which can be used to build the model."""

    model = keras.Sequential()

    # Create variables based on the desired keyword arguments used to build
    # the model. These must be changed in accordance with the **parameters.
    UNITS = parameters['UNITS']

    # Other parameters
    OPTIMIZER = 'adam'  # try out different optimizer (dynamic loss rate?)

    model.add(keras.layers.LSTM(UNITS, input_shape=(X_train.shape[1:])))
    model.add(keras.layers.Dense(2))

    model.compile(loss='mae', optimizer=OPTIMIZER)

    model.summary() # optional printout of key model properties

    return model

def train(model,X_train,y_train,X_test,y_test,**parameters):
    """Description."""

    # Create variables based on the desired keyword arguments used to train
    # the model. These must be changed in accordance with the **parameters.
    EPOCHS = parameters['EPOCHS']
    BATCH_SIZE = parameters['BATCH_SIZE']

    history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    shuffle=False
                )
    return model, history

def test_model(model,history):
    """Description."""
    performance = None

    return performance

if __name__ == '__main__':
    import sys, os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
