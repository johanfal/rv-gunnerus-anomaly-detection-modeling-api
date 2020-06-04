#-----------------------------------------------------------------------------
# File: modeling_funcs.py
# Purpose:
#   
#
# Created by: Johan Fredrik Alvsaker
# Last modified: 
#-----------------------------------------------------------------------------
# Standard library:
import pickle, sys

# External modules:
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Local API:
from src.api import file_management as filemag
from src.api import memory as mem
#-----------------------------------------------------------------------------

def reshape(
            df_train:pd.DataFrame,
            df_test:pd.DataFrame=None,
            output_cols:list=None,
            timesteps:int=1,
            verbose:bool=True
        ) -> [pd.DataFrame, pd.DataFrame]:
    """Description."""
    if df_test is not None: spec=' training'
    else: spec=''
    if verbose: print(f"Dimensionality of{spec} data: {df_train.shape}")
    X_train, y_train = _reshape_data(
                                        df_train,
                                        timesteps,
                                        output_cols=output_cols,
                                        bar_desc=f'Reshaping{spec} data..'
                                    )
    if verbose:
        print(f"Dimensionality of reshaped{spec} data: X_train: "\
                f"{X_train.shape} | y_train: {y_train.shape}.")
    if df_test is not None:
        if verbose:
            print(f"Dimensionality of test data: {df_test.shape}")
        X_test, y_test = _reshape_data(
                                        df_test,
                                        timesteps,
                                        output_cols=output_cols,
                                        bar_desc='Reshaping test data..'
                                    )
        if verbose: print("Dimensionality of reshaped testing data: X_test: "\
                            f"{X_test.shape} | y_test: {y_test.shape}.")
        return  [X_train, y_train, X_test, y_test]
    else: return X_train, y_train

def transform(
                data:pd.DataFrame,
                training_pct:float=0.8,
                scaler_type:str='minmax'
            ) -> ['sklearn.preprocessing.scaler', pd.DataFrame]:
    """Transforms a given set of data to normalized sets of training and
    testing data. The transformed values are returned as two dataframes,
    representing the training data and testing data, respectively."""

    # Create training and testing sets
    train_size = int(np.ceil(data.shape[0] * training_pct))

    df_train = data.iloc[:train_size]

    # Scaler
    scaler = get_scaler(scaler_type)
    scaler = scaler.fit(df_train[df_train.columns])
    arr_train = scaler.transform(df_train) # transformed training array

    # Add transformed arrays to training dataframe:
    df_train = pd.DataFrame(
                            arr_train,
                            columns=df_train.columns,
                            index=df_train.index
                        )
    if training_pct < 1.0:
        df_test = data.iloc[train_size:]
        arr_test = scaler.transform(df_test) # transformed testing array

        # Add transformed arrays to testing dataframe:
        df_test = pd.DataFrame(
                                arr_test,
                                columns=df_test.columns,
                                index=df_test.index
                            )
        return scaler, df_train, df_test
    elif training_pct > 1.0:
        sys.exit(f"Training percentage has value {training_pct}, "\
                    "but can not be above 1.0.")
    else: return scaler, df_train

def get_scaler(scaler_type:str='minmax') -> 'sklearn.preprocessing.scaler':
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
        # Scaler that normalizes about a zero-mean with unit variance
        return StandardScaler()

    if scaler_type == 'your_scaler_type_goes_here':
        return None # change to fit with your desired scaler

def get_df_hat(df:pd.DataFrame,y_hat:np.ndarray,prediction_cols:list=None
            ) -> pd.DataFrame:
    """Integrate values of predicted columns into existing dataframe. Through
    prediction_cols, the function replaces the corresponding columns in the
    input df with array values in y_hat (these must be ordered according to
    the prediction_cols)."""
    if prediction_cols == None: prediction_cols = df.columns
    df_hat = df.drop(prediction_cols, axis=1)

    pred_col_counter = 0
    for pred_col in prediction_cols:
        try:
            current_values = [row[pred_col_counter] for row in y_hat]
        except:
            sys.exit(
                "Mismatch between y_hat and prediction columns.\nIndex "\
                f"{pred_col_counter} corresponding with prediction column "\
                f"'{prediction_cols[pred_col_counter]}' is out of range for "\
                f" y_hat with shape ({y_hat.shape[0]},{y_hat.shape[1]})."
            )
        for i in range(df.columns.__len__()):
            if pred_col == df.columns[i]:
                df_hat.insert(loc=i,column=pred_col,value=current_values)
                break
        pred_col_counter += 1
    return df_hat

def inverse_transform_dataframe(
                                df:pd.DataFrame,
                                scaler:'sklearn.preprocessing.scaler'
                            ) -> pd.DataFrame:
    """Description."""
    itf_arr = scaler.inverse_transform(df)
    itf_df = pd.DataFrame(index=df.index)
    for i in range(df.columns.__len__()):
        itf_df[df.columns[i]] = [row[i] for row in itf_arr]

    return itf_df

def _reshape_data(
                    df:pd.DataFrame,
                    timesteps:int=1,
                    output_cols:list=None,
                    bar_desc:str=None
                ) -> [np.array, np.array]:
    """Reshapes a given dataframe to a 3D tensor based on the columns in the
    data (desired features), desired timesteps, and desired output columns
    (features to predict). The optional argument bar_desc is a description for
    the progress bar printed to the console."""

    Xs, ys = [], [] # empty placeholders
    range_max = df.shape[0] - timesteps # iteration range

    bar = filemag.get_progress_bar(range_max, bar_desc).start() # progress bar

    df_x = df

    # If the desired number of outputs (values to be predicted) is not equal
    # to the number of inputs (features), create a filtered dataframe:
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


def get_modelstring(prefix:str='model',**properties) -> str:
    """Create desired filename for storing models. Use **kwargs to specify
    the properties of the model by specifying keywords and their corresponding
    values. As an example, using 'units=64' will add 'units-64' to the
    filename, indicating that the model used 64 units. Best practice is to
    make the values dynamic, e.g. by using 'units=UNITS'."""
    modelstring = prefix
    for key, arg in properties.items():
        if type(arg) is int:
            arg = str(arg).zfill(3)
        modelstring = modelstring + f"_{str(key)}-{arg}"
    return modelstring

def get_anomaly_range(df_loss:pd.DataFrame,threshold:float,size:int=0
                    ) -> pd.DataFrame:
    """Description"""
    
    df_bool = df_loss > threshold
    return _check_neighboring_bools(df_bool,size=size).values

def _check_neighboring_bools(df_bool:pd.DataFrame,size:int=0
                        ) -> pd.DataFrame:
    """Checks if the neighborhood surrounding a boolean True value is also
    True. If not, the boolean value is changed to False. The function helps
    remove false outliers which will otherwise trigger unnwanted anomalies.
    The input variable 'size' defines the number of elements to include
    in the neighborhood of each timestep. If size=2n, timestep t will
    result in a neighborhood containing values t-n, t-n+1,...,t,...,t+n-1,t+n,
    thus yielding a neighborhood of 2n+1 elements (including t itself). If
    desired neighborhood size is odd with neighborhood=2n+1, the function will
    include an extra comparison value at a later timestep, resulting in the
    values t-n,...,t,...t+n+1."""

    df_nh = _get_neighborhood(df_bool, size)
    df_center_anoms = df_nh.all(axis='columns')
    df_center_anoms_nh = _get_neighborhood(df_center_anoms, size)

    return df_center_anoms_nh.any(axis='columns')


def _get_neighborhood(df_bool:pd.DataFrame,neighborhood:int=0) -> pd.DataFrame:
    df_neighborhood = pd.DataFrame(df_bool.values)
    for i in range(1, int(neighborhood/2)+1):
        df_neighborhood[f'-{i}'] = df_bool.shift(-i, fill_value=False).values
        df_neighborhood[f'+{i}'] = df_bool.shift(i, fill_value=False).values
    if neighborhood % 2:
        j = int((neighborhood+1)/2)
        df_neighborhood[f'+{j}'] = df_bool.shift(j,fill_value=False).values
    return df_neighborhood

def get_faulty(
                root_dir:str,
                cols:list,
                timesteps:int,
                index_col:'str'=None,
                chunksize:int=None,
                filter_operation:bool=True,
                faulty_suffix:str='faulty',
                action_parameters:list=[True]*3,
                scaler_type:str='minmax',
                output_cols:list=None,
            ) -> [pd.DataFrame,np.array,'sklearn.preprocessing.scaler']:
    """Description. Returns a complete dataframe in the format expected by the
    program, as well as the transformed, reshaped data alongside the
    transformation scaler, which is used for inverse transformation."""

    [create_data_file, do_transform, do_reshape] = action_parameters

    if create_data_file:
        complete_data = filemag.get_and_store_data(
                                            root_dir=root_dir,
                                            cols=cols,
                                            index_col=index_col,
                                            chunksize=chunksize,
                                            filter_operation=filter_operation,
                                            file_suffix=faulty_suffix,
                                            faulty_data=[]
                                        )
    else:
        complete_data = mem.load(file_suffix=faulty_suffix)

    if do_transform:
        [scaler, df_faulty,] = transform(
                                            complete_data,
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
        [X_reshaped,y_reshaped] = reshape(
                                        df_faulty,
                                        output_cols=output_cols,
                                        timesteps=timesteps
                                    )
        mem.store(
                    [X_reshaped,y_reshaped],
                    file_prefix='reshaped',
                    file_suffix=faulty_suffix
                )
    else:
        [X_reshaped,y_reshaped] = mem.load(
                                        file_prefix='reshaped',
                                        file_suffix=faulty_suffix
                                    )
    return df_faulty, X_reshaped, scaler

def get_absolute_error(df_real:pd.DataFrame,df_pred:pd.DataFrame) -> dict:
    """Calculate absolute error for each predicted timestep."""

    absolute_error = {}
    for col in df_pred.columns:
        # If values are predicted for a range of timesteps:
        if df_pred.shape.__len__() == 3:
            # Calculate as mean absolute error for each range:
            absolute_error[col] = np.mean(np.abs(df_real[col]-df_pred[col]),axis=1)
        else: # if values are predicted for one timestep at a time
            # Calculate as absolute error for each timestep:
            absolute_error[col] = np.abs(df_real[col] - df_pred[col])
    return absolute_error

def get_mae(absolute_error:dict) -> dict:
    """Calculate mean absolute error (MAE) from a dictionary of absolute_error
    values for each timestep of a predicted column. The absolute_error input
    is a dict with the predicted column as key and absolute_error values for
    each timestep as values. The MAE is returned as a dict as well with the
    mean of all timesteps as values for predicted column keys."""

    mae = {}
    for key, values in absolute_error.items():
        mae[key] = np.mean(values)
    return mae

def _get_uniform_thresholds(absolute_error:dict, threshold_pct:float) -> dict:
    """Calculate threshold values for all prediction columns based on a single
    desired threshold value."""

    thresholds = {}
    for key, values in absolute_error.items():
        thresholds[key] = np.percentile(values, threshold_pct)
    return thresholds

def _get_variable_thresholds(absolute_error:dict,threshold_pct:list) -> dict:
    """Calculate threshold values for all prediction columns based on a list
    of desired threshold values."""

    thresholds = {}
    counter = 0
    for key, values in absolute_error.items():
        thresholds[key] = np.percentile(values, threshold_pct[counter])
        counter += 1
    return thresholds

def get_thresholds(absolute_error:dict,threshold_pct) -> dict:
    if type(threshold_pct) is list:
        return _get_variable_thresholds(absolute_error,threshold_pct)
    else:
        return _get_uniform_thresholds(absolute_error,threshold_pct)

def get_performance(
                    df_pred_filtered:pd.DataFrame,
                    df_real_filtered:pd.DataFrame,
                    absolute_error:dict,
                    thresholds:dict,
                    anomaly_neighborhood:int=1
                ) -> dict:
    """DESCRIPTION."""

    performance = {}
    counter = 0
    # Iterate through each predicted column and add performance metrics:
    for col in df_pred_filtered.columns:
        performance[col] = pd.DataFrame(index=df_pred_filtered.index)
        performance[col]['pred'] = df_pred_filtered[col]
        performance[col]['real'] = df_real_filtered[col]
        performance[col]['loss'] = absolute_error[col]
        performance[col]['anom'] = get_anomaly_range(
                                                        performance[col].loss,
                                                        thresholds[col],
                                                        anomaly_neighborhood
                                                )
        counter += 1
        false_anomalies = get_false_anomalies(
                                                performance[col].loss,
                                                performance[col].anom,
                                                thresholds[col]
                                            )
        performance[col].drop(false_anomalies, inplace=True)
    return performance

def get_false_anomalies(
                        df_loss:pd.DataFrame,
                        df_true_anomalies:pd.DataFrame,
                        threshold:float
                    ) -> list:
    """DESCRIPTION."""
    all_anomalies = df_loss.index[df_loss > threshold].tolist()
    true_anomalies = df_true_anomalies.index[df_true_anomalies==True].tolist()
    # Return false anomalies as intersection between all and true anomalies:
    return np.setdiff1d(all_anomalies,true_anomalies)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')