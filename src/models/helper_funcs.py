import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.funcs.file_management import get_progress_bar


def transform(data,training_pct=0.8,normal_dist=False):
    """Transforms a given set of data to normalized sets of training and
    testing data. The transformed values are returned as two dataframes,
    representing the training data and testing data, respectively."""

    # Create training and testing sets
    train_size = int(np.ceil(data.shape[0] * training_pct))

    df_train, df_test = data.iloc[:train_size], data.iloc[train_size:]

    # Scaler
    if normal_dist:
        scaler = StandardScaler() # normalize about a zero-mean with unit variance
    else:
        scaler = MinMaxScaler(feature_range=(0,1)) # normalize values between 0 and 1

    scaler = scaler.fit(df_train[df_train.columns])

    arr_train = scaler.transform(df_train) # transformed training array
    arr_test = scaler.transform(df_test) # transformed testing array

    # Add transformed arrays to dataframe
    df_train = pd.DataFrame(arr_train, columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(arr_test, columns=df_test.columns, index=df_test.index)

    return scaler, df_train, df_test


def reshape_data(df,timesteps=1,output_cols=None,bar_desc=None):
    """Reshapes a given dataframe to a 3D tensor based on the columns in the
    data (desired features), desired timesteps, and desired output columns
    (features to predict). The optional argument bar_desc is a description for
    the progress bar printed to the console."""

    Xs, ys = [], [] # empty placeholders
    range_max = df.shape[0] - timesteps # iteration range

    bar = get_progress_bar(range_max, bar_desc).start() # progress bar

    df_x = df

    # If the desired number of outputs (values to be predicted) is not the same
    # as the number of inputs (features), create a filtered dataframe
    if output_cols is not None:
        df_y = df[output_cols]
    else:
        df_y = df

    for i in range(range_max):
        bar.update(i+1)
        Xs.append(df_x.iloc[i:(i + timesteps)].values) # add timesteps t-N to t-1 (i:(i+timesteps))
        ys.append(df_y.iloc[i + timesteps].values) # add timestep t (i+timesteps)
    bar.finish()
    return np.array(Xs), np.array(ys)

if __name__ == '__main__':
    import sys, os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
