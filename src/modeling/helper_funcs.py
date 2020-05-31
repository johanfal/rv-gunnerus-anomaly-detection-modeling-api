import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.funcs import file_management as filemag
from src.funcs import memory as mem


def reshape(df_train, df_test=None,output_cols=None,timesteps=1,verbose=True):
    """Description."""

    if verbose: print(f"Training data dimensionsality: {df_train.shape}")
    X_train, y_train = _reshape_data(
                                        df_train,
                                        timesteps,
                                        output_cols=output_cols,
                                        bar_desc='Reshaping training data..'
                                    )
    if verbose:
        print("Reshaped training data dimensionsality: X_train: "\
                f"{X_train.shape} | y_train: {y_train.shape}.")
    if df_test is not None:
        if verbose:
            print(f"Test data dimensionality: {df_test.shape}")
        X_test, y_test = _reshape_data(
                                        df_test,
                                        timesteps,
                                        output_cols=output_cols,
                                        bar_desc='Reshaping test data..'
                                    )
        if verbose: print("Reshaped testing data dimensionsality: X_test: "\
                            f"{X_test.shape} | y_test: {y_test.shape}.")
        return  [X_train, y_train, X_test, y_test]
    else: return X_train, y_train

def transform(data,training_pct=0.8,scaler_type='minmax'):
    """Transforms a given set of data to normalized sets of training and
    testing data. The transformed values are returned as two dataframes,
    representing the training data and testing data, respectively."""

    # Create training and testing sets
    train_size = int(np.ceil(data.shape[0] * training_pct))

    df_train, df_test = data.iloc[:train_size], data.iloc[train_size:]

    # Scaler
    scaler = get_scaler(scaler_type)
    scaler = scaler.fit(df_train[df_train.columns])

    arr_train = scaler.transform(df_train) # transformed training array
    arr_test = scaler.transform(df_test) # transformed testing array

    # Add transformed arrays to dataframe
    df_train = pd.DataFrame(
                            arr_train,
                            columns=df_train.columns,
                            index=df_train.index
                        )
    df_test = pd.DataFrame(
                            arr_test,
                            columns=df_test.columns,
                            index=df_test.index
                        )

    return scaler, df_train, df_test

def get_scaler(scaler_type='minmax'):
    """Normalize data based on a desired scaler. Default supported scalers are
    MinMaxScaler() and StandardScaler(). StandardScaler() is used if the data
    is known to have a normal distribution, while the MinMaxScaler() can be
    used otherwise. It is possible to implement other scalers if it is deemed
    necessary. If so, remember to import the appropriate scalers from
    sklearn.preprocessing, and add them as an option in the if statements
    below. Read the documentation on scikit's preprocessing page to learn more
    about possible scalers and their properties/use-cases."""

    if scaler_type == 'minmax':
        # Default scaler, which can suffer from the presence of large outliers
        feature_range = (0,1)
        return MinMaxScaler(feature_range=feature_range)
    if scaler_type == 'standard':
        # Data is known to have a normal distribution
        return StandardScaler() # normalize about a zero-mean with unit variance

    if scaler_type == 'your_scaler_type_goes_here':
        return None # change to fit with your desired scaler

def get_df_pred(df,y_hat,timesteps,prediction_cols=None):
    """Description."""
    if prediction_cols == None:
        prediction_cols = df.columns

    df_hat = df[timesteps:].drop(prediction_cols, axis=1)

    pred_col_counter = 0
    for pred_col in prediction_cols:
        current_values = [row[pred_col_counter] for row in y_hat]
        for i in range(df.columns.__len__()):
            if pred_col == df.columns[i]:
                df_hat.insert(loc=i,column=pred_col,value=current_values)
                continue
        pred_col_counter += 1

    for i in range(prediction_cols.__len__()):
        try:
            df_hat[prediction_cols[i]] = [row[i] for row in y_hat]
        except:
            sys.exit(
                "Mismatch between y_hat and prediction columns.\n" \
                f"Index {i} corresponding with prediction columns " \
                f"{prediction_cols[i]} is out of range for y_hat."
            )
    return df_hat

def inverse_transform_dataframe(df,scaler):
    """Description."""
    itf_arr = scaler.inverse_transform(df)
    itf_df = pd.DataFrame()
    for i in range(df.columns.__len__()):
        itf_df[df.columns[i]] = [row[i] for row in itf_arr]

    return itf_df

def _reshape_data(df,timesteps=1,output_cols=None,bar_desc=None):
    """Reshapes a given dataframe to a 3D tensor based on the columns in the
    data (desired features), desired timesteps, and desired output columns
    (features to predict). The optional argument bar_desc is a description for
    the progress bar printed to the console."""

    Xs, ys = [], [] # empty placeholders
    range_max = df.shape[0] - timesteps # iteration range

    bar = filemag.get_progress_bar(range_max, bar_desc).start() # progress bar

    df_x = df

    # If the desired number of outputs (values to be predicted) is not the same
    # as the number of inputs (features), create a filtered dataframe
    if output_cols is not None:
        df_y = df[output_cols]
    else:
        df_y = df

    for i in range(range_max):
        bar.update(i+1)

        Xs.append(
                df_x.iloc[i:(i + timesteps)].values # add timesteps t-N to t-1
            )
        ys.append(
                    df_y.iloc[i + timesteps].values # add timestep t
            )
    bar.finish()
    return np.array(Xs), np.array(ys)

def compare_models(models):
    """Idea of the function is to be able to view the different parameters
    and performance in e.g. a table. Could this be done in the web app?"""
    return

def get_modelstring(prefix='model',**kwargs):
    """Create desired filename for storing models. Use **kwargs to specify
    the properties of the model by specifying keywords and their corresponding
    values. As an example, using 'units=64' will add 'units-64' to the
    filename, indicating that the model used 64 units. Best practice is to
    make the values dynamic, e.g. by using 'units=UNITS'."""
    modelstring = prefix
    for key, arg in kwargs.items():
        if type(arg) is int:
            arg = str(arg).zfill(3)
        modelstring = modelstring + f"_{str(key)}-{arg}"
    return modelstring
    # return f"ts-{str(timesteps).zfill(3)}_ep-{str(EPOCHS).zfill(2)}_un-{str(UNITS).zfill(2)}_bs-{str(BATCH_SIZE).zfill(2)}"

def get_anomaly_range(df_loss,threshold,neighbors=0):
    """Description"""
    df_bool = df_loss > threshold
    return _check_neighboring_bools(df_bool,neighbors=neighbors)

def _check_neighboring_bools(df_bool,neighbors=0):
    """Checks if the neighborhood on each side of a boolean True value is also
    True. If not, the boolean value is changed to False. The function helps
    remove false outliers which will otherwise trigger unnwanted anomalies.
    The input variable 'neighbors' defines the number of values to include in
    the neighborhood of each timestep. If neighbors=n, timestep t will result
    in a neighborhood containing values t-n, t-n+1,...,t,...,t+n-1,t+n, thus
    yielding a neighborhood of 2n+1 elements (including t itself)."""
    df_neighborhood = pd.DataFrame(df_bool.values)
    for i in range(1, neighbors+1):
        df_neighborhood[f'{i}'] = df_bool.shift(-i,fill_value=False).values
        df_neighborhood[f'+{i}'] = df_bool.shift(i,fill_value=False).values
    return df_neighborhood.all(axis='columns').values

def get_faulty_reshaped(
                        root_dir,
                        cols,
                        timesteps,
                        index_col=None,
                        chunksize=None,
                        filter_operation=True,
                        faulty_suffix='faulty',
                        action_parameters=[True]*3,
                        scaler_type='minmax',
                        output_cols=None,
                    ):
    """Description."""

    [create_data_file, do_transform, do_reshape] = action_parameters

    if create_data_file:
        faulty_data = filemag.get_and_store_data(
                                            root_dir=root_dir,
                                            cols=cols,
                                            index_cols=index_col,
                                            chunksize=chunksize,
                                            filter_operation=filter_operation,
                                            file_suffix=faulty_suffix,
                                            faulty_data=[]
                                        )
        mem.store(faulty_data,file_suffix=faulty_suffix)
    else:
        faulty_data = mem.load(file_suffix=faulty_suffix)

    if do_transform:
        [scaler, df_faulty,] = transform(
                                                    faulty_data,
                                                    training_pct=1.0,
                                                    scaler_type=scaler_type
                                            )
        mem.store(
                    [scaler,df_faulty],
                    file_prefix='transformed',
                    file_suffix=faulty_suffix
                )
    else:
        [scaler,df_faulty] = mem.load(
                                        file_prefix='transformed',
                                        file_suffix=faulty_suffix
                                    )

    if do_reshape:
        [X_faulty,y_faulty] = reshape(
                                        df_faulty,
                                        output_cols=output_cols,
                                        timesteps=timesteps
                                    )
        mem.store(
                    [X_faulty,y_faulty],
                    file_prefix='reshaped',
                    file_suffix=faulty_suffix
                )
    else:
        [X_faulty,y_faulty] = mem.load(
                                        file_prefix='reshaped',
                                        file_suffix=faulty_suffix
                                    )
    return X_faulty, y_faulty

if __name__ == '__main__':
    import sys, os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')