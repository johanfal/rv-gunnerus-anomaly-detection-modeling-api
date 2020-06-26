# ----------------------------------------------------------------------------
# File: model_example_lstm.py
# Purpose: serve as an example of a machine learning model. The example file
# is implemended as a recurrent neural network (RNN), more specifically using
# a long short term memory (LSTM) approach. The model and layer properties are
# set to be as simple as possible, both to demonstrate the power of LSTM
# models for anomaly detection, but also t be easy to understand.
# Created by: Johan Fredrik Alvsaker
# Last modified: 26.6.2020
# ----------------------------------------------------------------------------
# Standard library:
import os
import pickle
import sys

# External modules:
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

# Local API:
from src.api import file_management as filemag
from src.api import memory as mem
from src.api import modeling_funcs as mfnc
from src.api import plotting_funcs as pfnc
# ----------------------------------------------------------------------------


def create(
        input_shape: tuple,
        verbose: bool = True,
        **parameters) -> 'keras.model':
    """Create the Keras model, which in this case is of type LSTM. The
    function takes a variable number of keyword arguments, which can be used
    to build the model. Change verbose to False to suppress model summary
    printout."""

    model = keras.Sequential()  # instantiate a sequential model

    # Create variables based on the desired keyword arguments used to build
    # the model. These must be changed in accordance with the **parameters.
    # (It is optional to use keyword arguments through **parameters.)
    UNITS_LSTM = parameters['UNITS_LSTM']  # DESCRIPTION
    DROPOUT_RATE = parameters['DROPOUT_RATE']
    DENSE_UNITS = parameters['UNITS_DENSE']
    # Add LSTM layer:
    model.add(layers.LSTM(UNITS_LSTM, input_shape=(input_shape)))

    # Add dropout layer to prevent overfitting:
    model.add(layers.Dropout(DROPOUT_RATE))

    # Add densely-connected neural network layer:
    model.add(keras.layers.Dense(DENSE_UNITS))

    # Compile model using MAE (mean absolute error), adam optimizer
    # (stochastic gradient descent algorithm), and accuracy metrics:
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    if verbose:
        model.summary()  # optional printout of key model properties

    return model


def train(
    model: 'keras.model',
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **parameters
) -> ['keras.model', list]:
    """Train model based on training paramters."""

    # Create variables based on the desired keyword arguments used to train
    # the model. These must be changed in accordance with the **parameters.
    # (It is optional to use keyword arguments through **parameters.)
    EPOCHS = parameters['EPOCHS']  # cycles going through training data
    BATCH_SIZE = parameters['BATCH_SIZE']  # repetitions before updating model

    # Optional checkpoint to only save the best model:
    checkpoint = ModelCheckpoint(
        'model_checkpoint.h5',
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
        mode='auto')

    # Fit data and retrieve training history:
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        shuffle=False,
                        # callbacks=[checkpoint]
                        )
    return model, history.history


def test(
    model: 'keras.model',
    history: dict,
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    threshold_pct: float,
    anomaly_neighborhood: int,
    pred_scaler: 'sklearn.preprocessing.scaler',
    test_scaler: 'sklearn.preprocessing.scaler' = None,
    prediction_cols: list = None,
    **parameters
) -> pd.DataFrame:
    """Test the fitted model by creating a performance file of different
    metrics. The performance file is used to visualize the results of training
    the model."""

    if test_scaler is None:
        test_scaler = pred_scaler

    # Predict values on testing data:
    y_hat = model.predict(X_test)

    # Retrieve and remove timestep difference (timesteps) from df_test:
    timesteps = df_test.shape[0] - y_hat.shape[0]
    df_test = df_test[timesteps:]

    # Merge original and predicted values in a single dataframe
    df_hat = mfnc.get_df_hat(df_test, y_hat, prediction_cols=prediction_cols)

    # Inverse transform df_hat and df_test back to original magnitudes:
    df_hat_inv = mfnc.inverse_transform_dataframe(df_hat, pred_scaler)
    df_test_inv = mfnc.inverse_transform_dataframe(df_test, test_scaler)

    # Filter prediction columns:
    df_hat_filtered = df_hat_inv.filter(prediction_cols)
    df_test_filtered = df_test_inv.filter(prediction_cols)

    # Calculate absolute error for each predicted timestep:
    absolute_error = mfnc.get_absolute_error(
        df_hat_filtered, df_test_filtered)

    # Calculate thresholds based on an anomaly distribution percentage:
    # (If threshold_pct is 70 %, the threshold value will be the minimum of
    # the 30 % highest values. Threshold_pct can either be passed as a single,
    # uniform value, or as a list of percentages for each predicted column.)
    # thresholds = mfnc.get_thresholds(absolute_error, threshold_pct)
    thresholds = mfnc.get_thresholds(absolute_error, threshold_pct)

    # Calculate mean absolute error (MAE) for each predicted column:
    mae = mfnc.get_mae(absolute_error)
    # Calculate root mean square error (RMSE) for each predicted column:
    # rmse = mfnc.get_rmse(df_hat_filtered, df_test_filtered)

    performance = mfnc.get_performance(
        df_pred_filtered=df_hat_filtered,
        df_real_filtered=df_test_filtered,
        absolute_error=absolute_error,
        thresholds=thresholds,
        anomaly_neighborhood=anomaly_neighborhood
    )

    return performance, absolute_error, thresholds


def visualize(performance: dict, history: dict, **kwargs) -> None:
    """Visualize modeling results through defined visualization functions. It
    is suggested to also implement other plots here if deemed necessary."""

    # Retrieving parameters:
    absolute_error = kwargs['absolute_error']
    thresholds = kwargs['thresholds']
    units = kwargs['units']

    # Training history:
    pfnc.get_historyplt(history)
    # (if modeling checkpoint is used, the history plot will not be as
    # descriptive, as the plot will only update values if a successive epoch
    # performed better.)
    # Plot results for each predicted state:
    for key in performance:

        # # Predicted state values:
        pfnc.get_predplt(
            real=performance[key].real,
            pred=performance[key].pred,
            signal=key,
            unit=units[key]
        )

        # Loss distribution as 2x2 histograms with kernal density plots (KDE):
        pfnc.get_distplt(
            signal=key,
            dist=performance[key].loss,
            threshold=thresholds[key],
            unit=units[key]
        )

        # State value loss and threshold:
        pfnc.loss_threshold(
            signal=key,
            loss=performance[key].loss,
            threshold=thresholds[key],
            unit=units[key]
        )

        # Plot anomalies:
        pfnc.get_anomalyplt(
            signal=key,
            real=performance[key].real,
            anoms=performance[key].anom,
            unit=units[key]
        )

# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
