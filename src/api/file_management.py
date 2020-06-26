# ----------------------------------------------------------------------------
# File: file_management.py
# Purpose: handle file management mostly related to raw csv-files.
#
# Created by: Johan Fredrik Alvsaker
# Last modified: 26.6.2020
# ----------------------------------------------------------------------------
# Standard library:
import datetime
import json
import os
import sys
import time

# External modules:
import matplotlib.pyplot as plt
import pandas as pd
import progressbar
import win32file

# Local API:
from src.api import memory as mem
# ----------------------------------------------------------------------------


def check_access() -> None:
    """Check access to network drive with transmitted signal data. The
    function assumes that the user is only connected to one network drive."""

    if sys.platform == 'win32':
        print('Checking access to network drive..')
        DL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        drives = [
            '%s:' %
            d for d in DL if os.path.exists(
                '%s:' %
                d) and win32file.GetDriveType(
                '%s:' %
                d) == win32file.DRIVE_REMOTE]
        if(not drives):
            sys.exit("Error: access denied. "
                     "Remember to connect to the network drive and verify"
                     "that you have access to the local network folder.")
        else:
            print('Access granted!')
            return drives[0]
    else:
        raise NotImplementedError(
            f"Operating system {sys.platform} is not "
            "supported in the current implementation."
        )
        sys.exit()


def get_system_list(sensor: str, component: str = None) -> dict:
    """Returns a dictionary of systems and units from specified columns in the
    'rvgunnerus_systems.json' file."""

    with open('datastore/systems/rvgunnerus_systems.json', 'r') as f:
        json_columns = json.load(f)
    if not component:
        return json_columns['sensors'][sensor]
    else:
        return json_columns['sensors'][sensor][component]


def get_single_day_dir(
    network_location: str,
    sensor: str,
    year: int,
    month: int,
    date: int
) -> str:
    """Get the directory location for a desired sensor at a desired date"""

    return os.path.join(
        network_location, sensor,
        str(year),
        str(month).zfill(2),
        str(date).zfill(2)
    )


def get_file_list(file_dir: str) -> list:
	"""Returns a list of files from a given directory."""
	try:
		return os.listdir(file_dir)
	except ValueError:
		print('Either a list of files or file directory must be provided.')
		sys.exit('Adjust user-specified input.')


def concatenate_files(file_dir: str,
                      cols: list = None,
                      index_col: str = None,
                      chunksize: int = None,
                      filter_operation: bool = True,
                      faulty_data: list = []
                      ) -> pd.DataFrame:
    """Concatenate dataframes from imported csv.gz-format."""
    cols.insert(0, index_col) # add index column
    dfs = []  # list for storing dataframes to be concatenated

    # Get list of files to be concatenated from the desired file directory:
    file_list = get_file_list(file_dir)

    # Filter data where the vessel is not in operation:
    filter_column = 'ME1_EngineSpeed'
    FILTER_VALUE = 1770  # speed when main engine 1 is in operation

    # Faulty data
    if faulty_data.__len__() != 0:
        faulty_dict = {}
        for interval in faulty_data:
            faulty_dict[interval[0]] = [interval[1], interval[2]]

    # Iterate through files in desired file list
    for file in file_list:

        # Create a dataframe for current file with designated reading options
        df = pd.read_csv(file_dir + '\\' + file,
                         compression='gzip',
                         header=0,
                         sep=',',
                         usecols=cols,
                         parse_dates=['time'],
                         date_parser=_date_parser,
                         chunksize=chunksize,
                         error_bad_lines=True)
        df = df.set_index(index_col) # set index column

        # Apply operation filter if specified by function input
        if(filter_operation):  # filter based on appropriate measures
            df = df[df[filter_column] > FILTER_VALUE]

        if faulty_data.__len__() != 0:  # remove simulated error interval
            if file in faulty_dict:
                simulated_error_start = faulty_dict[file][0]
                simulated_error_end = faulty_dict[file][1]
                df = remove_faulty_data(
                    df,
                    simulated_error_start,
                    simulated_error_end,
                    file
                )

        dfs.append(df)  # append dataframe from file to the list of dataframes

    # If time is not specified as index column, ignore numbering indices in
    # concatenated dataframe:
    if index_col is None:
        ignore_index = True
    else:
        ignore_index = False  # keep indexing if time is index column
    return concatenate_dataframes(dfs, index_col)  # concatenate dataframes


def get_startpath(
    network_location: str,
    sensor: str = None,
    training_period: list = [None] * 3
) -> str:
    """"Return string with starting directory path based on desired data
    resolution."""

    [year, month, day] = training_period
    startpath = network_location
    for item in [sensor, year, month, day]:
        if item is not None:
            startpath = os.path.join(startpath, str(item))
        else:
            break
    return startpath


def get_data(
    root_dir: str,
    cols: list,
    index_col: str = None,
    chunksize: int = None,
    filter_operation: bool = False,
    faulty_data: list = [],
    verbose: bool = True
) -> pd.DataFrame:
    """Retrieve a concatenated dataframe with all data from the given root
    directory."""

    dfs = []  # list for storing dataframes to be concatenated
    if verbose:
        print(f"Retrieving data from '{root_dir}'...")
        n_files = 0  # number of files concatenated
        n_dirs = 1
    # Iterate through file hierarchy based on the given root directory
    for root, dirs, files in os.walk(root_dir):

        # If list of files at current directory is not empty, concatenate
        # contained files:
        if dirs.__len__() > 0:
            n_dirs = dirs.__len__()
            if verbose:
                print(f"Directories in '{root}': {dirs.__len__()}.")

        if files != []:
            if verbose:
                print(f"Current directory: {root} ({files.__len__()} files)."
                      f" {n_files} of {n_dirs} directories appended from "
                      f"parent directory.", end='\r')
                n_files += 1
            # Concatenate files in given directory based on specified options
            df = concatenate_files(file_dir=root,
                                   cols=cols,
                                   index_col=index_col,
                                   chunksize=chunksize,
                                   filter_operation=filter_operation,
                                   faulty_data=faulty_data
                                   )
            dfs.append(df)  # Append current dataframe to list of dataframes
            if verbose and n_files == n_dirs:
                print(f"Current directory: {root} ({files.__len__()} files)."
                      f" {n_dirs} of {n_dirs} directories appended from "
                      f"parent directory.", end='\r')
    if verbose:
        print("")
    return concatenate_dataframes(dfs, index_col)  # concatenate dataframes


def get_and_store_data(
    root_dir: str,
    cols: list,
    index_col: str = None,
    chunksize: int = None,
    filter_operation: bool = False,
    file_suffix='store',
    faulty_data: list = [],
    verbose: bool = True
) -> pd.DataFrame:
    """Calls the 'get_data()' function, as well as storing the data, in its
	entirety, is non-empty. The data is stored through the memory.py function
	'store()'."""

    data = get_data(
        root_dir=root_dir,
        cols=cols,
        index_col=index_col,
        chunksize=chunksize,
        filter_operation=filter_operation,
        faulty_data=faulty_data,
        verbose=verbose
    )

    if data.empty:  # all data in time interval is filtered as non-operational
        sys.exit(
            'No data in the selected interval qualifies the filtering '
            'conditions. No object has been pickled to memory.'
        )
    else:
        mem.store(data, file_suffix=file_suffix)  # store data as file
    return data


def concatenate_dataframes(dfs, index_col=None):
    """Concatenate list of dataframes. If time is not specified as index
    column through the index_col variable, the indexed numbering is ignored in
    the concatenated dataframe."""

    if index_col is None:
        ignore_index = True
    else:
        ignore_index = False
    return pd.concat(dfs, axis=0, ignore_index=ignore_index)


def all_equal(values: list, val) -> bool:
    """Returns boolean value corresponding to all values in a list array being
    equal."""

    return all(elem == val for elem in list)


def get_datetime(filename: str) -> datetime.time:
    """Returns a datetime.time() object from a filename in known format."""

    time_string = filename[:filename.rfind('_')]
    time_string = time_string[time_string.rfind('_') + 1:]
    year = int(time_string[0:4])
    month = int(time_string[4:6])
    day = int(time_string[6:])
    return datetime.datetime(year=year, month=month, day=day).time()


def to_datetime(time_string: str, format: str = '%Y-%m-%d %H:%M:%S.%f'
                ) -> datetime.time:
    """Returns a datetime.time() object from a string in known format."""

    return datetime.datetime.strptime(time_string, format).time()

    remove_faulty_data(1,)


def remove_faulty_data(df: pd.DataFrame, start: str, end: str, filename: str
                       ) -> pd.DataFrame:
    """Removes faulty data at a known time interval specified through a
    starting time and ending time. The input time values can either be in a
    known string format or as datetime.time() objects."""

    if(type(start) == str):
        start = to_datetime(start)
    if(type(end) == str):
        end = to_datetime(end)

    # Remove faulty data from df:
    # (Using end time before start time excludes the interval in-between)
    df_filtered = df.between_time(
        start_time=end,
        end_time=start,
        include_start=False,
        include_end=False
    )
    # Number of deleted rows due to faulty data:
    n_deleted_rows = df.shape[0] - df_filtered.shape[0]
    print(
        f"{n_deleted_rows} rows of faulty data between {start} and {end} "
        f"succesfully removed from '{filename}'.")
    return df_filtered


def get_progress_bar(range_max: int, bar_desc: str = None
                     ) -> progressbar.ProgressBar:
    """"Returns an object representing a progress bar according to the
    progressbar module."""
    if bar_desc:
        print(f'{bar_desc}'.format())
    return progressbar.ProgressBar(
        maxval=range_max, widgets=[
            progressbar.Bar(
                '=', '[', ']'), ' ', progressbar.Percentage()])


def _date_parser(time):
	"""Parser to convert string object to datetime object when reading
	csv-files using pandas."""

	return pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f')


if __name__ == '__main__':
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
