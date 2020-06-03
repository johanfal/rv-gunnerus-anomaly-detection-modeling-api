import pickle

import numpy as np
import pandas as pd

from src.funcs import memory as mem
from src.modeling import helper_funcs as fnc
from tensorflow import keras

def create(input_shape:tuple,verbose:bool=True,**parameters) -> 'keras.model':
    """DESCRIPTION. The function takes a variable number of keyword arguments,
    which can be used to build the model. Change verbose to false to suppress
    model summary printout."""

    model = keras.Sequential()

    # Create variables based on the desired keyword arguments used to build
    # the model. These must be changed in accordance with the **parameters.
    # (It is optional to use keyword arguments through **parameters.)
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
            model:'keras.model',
            X_train:pd.DataFrame,
            y_train:pd.DataFrame,
            X_test:np.ndarray,
            y_test:np.ndarray,
            **parameters
        ) -> ['keras.model', list]:
    """DESCRIPTION."""

    # Create variables based on the desired keyword arguments used to train
    # the model. These must be changed in accordance with the **parameters.
    # (It is optional to use keyword arguments through **parameters.)
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
            model:'keras.model',
            history:list,
            df_test:pd.DataFrame,
            X_test:np.ndarray,
            threshold_pct:float,
            anomaly_neighborhood:int,
            pred_scaler:'sklearn.preprocessing.scaler',
            test_scaler:'sklearn.preprocessing.scaler'=None,
            prediction_cols:list=None,
            **parameters
    ) -> pd.DataFrame:
    """DESCRIPTION."""
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

    # Filter prediction columns:
    df_hat_filtered = df_hat_inv.filter(prediction_cols)
    df_test_filtered = df_test_inv.filter(prediction_cols)

    # Calculate absolute error for each predicted timestep:
    absolute_error = fnc.get_absolute_error(df_hat_filtered, df_test_filtered)




    threshold_pcts = [
        # 90,
        # 91,
        # 92,
        # 93,
        # 94,
        95,
        96,
        97,
        # 97.10,
        # 97.20,
        # 97.25,
        # 97.30,
        # 97.40,
        # 97.50,
        # 97.60,
        # 97.70,
        # 97.75,
        # 97.80,
        # 97.90,
        98,
        # 98.25,
        # 98.5,
        # 98.75,
        99,
    ]
    neighbors = [
        # 5,
        # 10,
        # 12,
        # 14,
        15,
        # 16,
        # 17,
        # 18,
        # 19,
        20,
        25,
        30,
        35,
        40
    ]
    result = {}
    for pct in threshold_pcts:
        for neighbor in neighbors:
            print(f"pct: {pct}. neighborhood: {neighbor}.")

            # Calculate thresholds based on an anomaly distribution percentage:
            # (If threshold_pct is 70 %, the threshold value will be the minimum of
            # the 30 % highest values. Threshold_pct can either be passed as a single,
            # uniform value, or as a list of percentages for each predicted column.)
            # thresholds = fnc.get_thresholds(absolute_error, threshold_pct)
            thresholds = fnc.get_thresholds(absolute_error, pct)

            # Calculate mean absolute error (MAE) for each predicted column:
            mae = fnc.get_mae(absolute_error)
            # Calculate root mean square error (RMSE) for each predicted column:
            # rmse = fnc.get_rmse(df_hat_filtered, df_test_filtered)
            performance = fnc.get_performance(
                                            df_pred_filtered=df_hat_filtered,
                                            df_real_filtered=df_test_filtered,
                                            absolute_error=absolute_error,
                                            thresholds=thresholds,
                                            # anomaly_neighborhood=anomaly_neighborhood
                                            anomaly_neighborhood=neighbor
                                        )
            result[f"pct-{pct}_neg-{neighbor}"] = [performance,absolute_error,thresholds]
    mem.store(result,file_prefix='faulty_testing_results')
    import sys
    sys.exit("Done.")
    # Necessary parameters:
    # col, absolute_error, thresholds, df_hat_filtered, df_test_filtered,
    return performance, absolute_error, thresholds

def visualize(performance:dict,history:list, **kwargs) -> None:

    absolute_error = kwargs['absolute_error']
    thresholds = kwargs['thresholds']


    for key, df in performance.items():
        print("hello!")
    return

if __name__ == '__main__':
    import sys, os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
