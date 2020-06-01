import pickle

import numpy as np
import pandas as pd

from src.funcs import memory as mem
from src.modeling import helper_funcs as fnc
from tensorflow import keras

def create(input_shape:tuple,verbose:bool=True,**parameters) -> keras.model:
    """Description. The function takes a variable number of keyword arguments,
    which can be used to build the model. Change verbose to false to suppress
    model summary printout."""

    model = keras.Sequential()

    # Create variables based on the desired keyword arguments used to build
    # the model. These must be changed in accordance with the **parameters.
    UNITS = parameters['UNITS']
    OPTIMIZER = 'adam'  # try out different optimizer (dynamic loss rate?)

    model.add(keras.layers.LSTM(UNITS, input_shape=(input_shape)))
    model.add(keras.layers.Dense(2))
    # model.add(keras.layers.Dropout(rate=DROPOUT_RATE))
    # model.add(keras.layers.Dense(2, input_dim = X_train.shape[1]))
    # model.add(keras.layers.RepeatVector(n=1)) # This was the last thing you changed, and it seemed to be working..
    # # ! Is it possible to replace the hard-coded value of 1 with the shape of y_train, somehow?
    # model.add(keras.layers.LSTM(UNITS, return_sequences=True))
    # model.add(keras.layers.Dropout(rate=DROPOUT_RATE))
    # # model.add(keras.layers.Dense(2))
    # model.add(keras.layers.TimeDistributed(
    # keras.layers.Dense(2)) # change this to use the shape of y_train
    # )

    model.compile(loss='mae', optimizer=OPTIMIZER)
    if verbose:
        model.summary() # optional printout of key model properties

    return model

def train(
            model:keras.model,
            X_train:pd.DataFrame,
            y_train:pd.DataFrame,
            X_test:np.ndarray,
            y_test:np.ndarray,
            **parameters
        ) -> [keras.model, list]:
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
    return model, history.history

def test(
            model:keras.model,
            history:list,
            df_test:pd.DataFrame,
            X_test:np.ndarray,
            threshold_pct:float,
            pred_scaler:sklearn.preprocessing.scaler,
            test_scaler:sklearn.preprocessing.scaler=None,
            prediction_cols:list=None,
            **parameters
    ) -> pd.DataFrame:
    """Description."""
    # THRESHOLD_PCT = parameters['THRESHOLD_PCT']
    # ANOMALY_NEIGHBORS = parameters['ANOMALY_NEIGHBORS']

    if test_scaler is None: test_scaler=pred_scaler

    # Predict values on testing data:
    y_hat = model.predict(X_test)

    # Retrieve and remove timestep difference (timesteps) from df_test:
    timesteps = df_test.shape[0] - y_hat.shape[0]
    df_test = df_test[timesteps:]

    # Merge original and predicted values in a single dataframe
    df_hat = fnc.get_df_hat(df_test,y_hat,prediction_cols=prediction_cols)

    # Inverse transform df_hat and df_test back to original magnitudes:
    df_hat_inv = fnc.inverse_transform_dataframe(df_hat, pred_scaler)
    df_test_inv = fnc.inverse_transform_dataframe(df_test, test_scaler)

    # Filter prediction columns(remove?):
    df_hat_filtered = df_hat_inv.filter(prediction_cols)
    df_test_filtered = df_test_inv.filter(prediction_cols)

    # Calculate absolute error for each predicted timestep:
    absolute_error = fnc.get_absolute_error(df_hat_filtered, df_test_filtered)


    thresholds = fnc.get_thresholds(absolute_error)

    # Calculate mean absolute error (MAE) for each predicted column:
    mae = fnc.get_mae(absolute_error)
    # Calculate root mean square error (RMSE) for each predicted column:
    # rmse = fnc.get_rmse(df_hat_filtered, df_test_filtered)

    performance = None

    return performance

if __name__ == '__main__':
    import sys, os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
