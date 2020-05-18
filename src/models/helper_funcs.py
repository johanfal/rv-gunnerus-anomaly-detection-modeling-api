import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.funcs.file_management import get_progress_bar

def transform(data, training_pct=0.8):
    """Description."""

    # Create training and testing sets
    train_size = int(np.ceil(data.shape[0] * training_pct))

    df_train, df_test = data.iloc[:train_size], data.iloc[train_size:]

    # Scaler
    scaler = StandardScaler()
    scaler = scaler.fit(df_train[df_train.columns])

    arr_train = scaler.transform(df_train) # transformed training array
    arr_test = scaler.transform(df_test) # transformed testing array

    # Add transformed arrays to dataframe
    df_train = pd.DataFrame(arr_train, columns=df_train.columns, index=df_train.index)
    df_test = pd.DataFrame(arr_test, columns=df_test.columns, index=df_test.index)

    return df_train, df_test


def reshape_data(df, timesteps = 1, bar_desc=None):
    """Description."""

    Xs, ys = [], [] # empty placeholders
    range_max = df.shape[0] - timesteps # iteration range

    bar = get_progress_bar(range_max, bar_desc).start() # progress bar

    for i in range(range_max):
        bar.update(i+1)
        Xs.append(df.iloc[i:(i + timesteps)].values)
        ys.append(df.iloc[i + timesteps])
    bar.finish()
    return np.array(Xs), np.array(ys)

if __name__ == '__main__':
    import sys
    sys.exit('Run from manage.py, not model.')
