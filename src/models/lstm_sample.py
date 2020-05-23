import pickle

import numpy as np
import pandas as pd

from src.funcs import memory as mem
from src.models import helper_funcs as fnc
from tensorflow import keras

def create(data,
            prediction_cols=None,
            training_pct=0.8,
            timesteps=100,
            create_data_file=True,
            do_transform=True,
            normal_dist=False,
            do_reshape=True,
            delete_pickled_files=True
        ):

    if prediction_cols is None:
        prediction_cols = data.columns


    LOAD_RESHAPED = 'reshaped_complete_data_set' # ! REMOVE or find alternative
    LOAD_TRANSFORMED = 'transformed_complete_data_set' # ! REMOVE or find alternative

    if do_transform:
        scaler, df_train, df_test = fnc.transform(data, training_pct, normal_dist=normal_dist)
        mem.store([scaler, df_train, df_test], 'transformed')
    else:
        [scaler, df_train, df_test] = mem.load(filename=LOAD_TRANSFORMED)
    if do_reshape:
        print(f"Training data dimensionsality: {df_train.shape}")
        X_train, y_train = fnc.reshape_data(df_train,timesteps,output_cols=prediction_cols,bar_desc='Reshaping training data..')
        print(f"Reshaped training data dimensionsality: X_train: {X_train.shape} | y_train: {y_train.shape}.")
        print(f"Test data dimensionality: {df_test.shape}")
        X_test, y_test = fnc.reshape_data(df_test,timesteps,output_cols=prediction_cols,bar_desc='Reshaping test data..')
        print(f"Reshaped testing data dimensionsality: X_test: {X_test.shape} | y_test: {y_test.shape}.")
        print("Storing reshaped dataframes as 'src/datastore/reshaped.pckl'.")
        reshaped = mem.store([X_train, y_train, X_test, y_test], 'reshaped')
        print("Data succesfully reshaped and stored.")
    else:
        [X_train, y_train, X_test, y_test] = mem.load(filename=LOAD_RESHAPED)


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

    UNITS = 128
    RETURN_SEQUENCE = True
    RATE = 0.2
    EPOCHS = 30
    BATCH_SIZE = 128
    VALIDATION_SPLIT = 0.1
    SHUFFLE = False

    # NEW MODEL
    model.add(keras.layers.LSTM(UNITS, input_shape=(X_train.shape[1:])))
    model.add(keras.layers.Dropout(rate=RATE))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.TimeDistributed(
    keras.layers.Dense(2))
    ) # 12 output variables
    # model.add(keras.layers.Dense(y_train.shape[1])) # 12 output variables


    model.compile(loss='mae', optimizer='adam')
    model.summary()

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=2, shuffle=False)

    MODELSTRING = f"ts-{str(timesteps).zfill(3)}_ep-{str(EPOCHS).zfill(2)}_un-{str(UNITS).zfill(2)}_bs-{str(BATCH_SIZE).zfill(2)}"

    mem.save_model(model, history, modelstring=MODELSTRING)

    y_hat = model.predict(X_test)


    # Need to inverse transform data

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

    return model


if __name__ == '__main__':
    import sys
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
