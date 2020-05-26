import pickle
import sys, os
from datetime import datetime

def store(
            obj,
            store_dir='src/datastore/',
            file_prefix=None,
            file_suffix=None
        ):
    """Description."""

    if file_prefix is None: file_prefix = 'store'
    store_dir = _check_dir_string(store_dir)
    filename = _get_filename(file_prefix, file_suffix)

    f = open(f'{store_dir}{filename}.pckl', 'wb')
    pickle.dump(obj, f)
    f.close()
    print(f"Object succesfully stored as '{filename}.pckl' in '{store_dir}'")
    return

def store_time_interval(
                        start,end,
                        store_dir='src/datastore/',
                        file_prefix=None,
                        file_suffix=None
                    ):
    """Description."""

    if file_prefix is None: file_prefix = 'store'

    store_dir = _check_dir_string(store_dir)
    filename = _get_filename(file_prefix, file_suffix) + '_timeint'

    f = open(f'{store_dir}{filename}.pckl', 'wb')
    obj = []
    for timestamp in [start, end]:
        date_string = str(timestamp.day) + '-' + str(timestamp.month) + '-'+ str(timestamp.year)
        obj.append(date_string)
    pickle.dump(obj, f)
    f.close()
    print(f"Object succesfully stored as '{filename}.pckl' in '{store_dir}'")
    return

def load(load_dir='src/datastore/',file_prefix=None,file_suffix=None):
    """Description."""
    if file_prefix is None: file_prefix = 'store'
    load_dir = _check_dir_string(load_dir)
    filename = _get_filename(file_prefix, file_suffix)

    try:
        f = open(f'{load_dir}{filename}.pckl', 'rb')
        obj = pickle.load(f)
        f.close()
        print(f"Object succesfully loaded from '{filename}' in '{load_dir}'")
        return obj
    except FileNotFoundError:
        sys.exit(f"{filename}.pckl not found in '{load_dir}'.")

def load_time_interval(
                        load_dir='src/datastore/',
                        file_prefix=None,
                        file_suffix=None
                    ):
    """Loads meta file for an accompanying data file. The meta file describes
    the time interval of the corresponding dataset."""

    if file_prefix is None: file_prefix = 'store'
    load_dir=_check_dir_string(load_dir)
    filename = _get_filename(file_prefix,file_suffix)
    return load(load_dir=load_dir,file_prefix=filename,file_suffix='timeint')

def delete(delete_dir='src/datastore/',file_prefix=None, file_suffix=None):
    """Description."""

    if file_prefix is None: file_prefix = 'store'
    delete_dir = _check_dir_string(delete_dir)
    filename = _get_filename(file_prefix, file_suffix)
    path = f'{delete_dir}{filename}.pckl'
    print(path)
    try:
        os.remove(path)
        print(f"Object stored in '{filename}.pckl' deleted from '{delete_dir}'")
    except:
        print(f"File '{filename}.pckl' not found in '{delete_dir}'.")
    return

def save_model(
                model,
                history,
                model_dir='src/datastore/models',
                file_prefix=None,
                modelstring='unspecificed'
            ):
    """Saves a Keras model to a file named after important properties and
    tuning parameters of the model. Each saved file receives a unique name
    based on the time of save. Thus, no models are overwritten.
    Suggested elements in kwargs:
        rms or loss of some sort
        units
        epochs
        datasize (reshaped, including features)
        number of outputs
        timesteps
    """

    if file_prefix is None: file_prefix = 'model'
    json_model = model.to_json()
    model_dir = _check_dir_string(model_dir)
    unique_time = datetime.now().strftime('%Y%m%d-%H%M')
    filename = f"{model_dir}{file_prefix}_{unique_time}_{modelstring}"
    f = open(f"{filename}.pckl", 'wb')
    pickle.dump([json_model, history.history], f)
    f.close()
    return

def load_model(
                file_prefix=None,
                model_dir='src/datastore/models/',
                file_suffix=None
            ):
    """Load a model and history from a local directory, specified through
    a model directory, filename prefix and filename suffix. The suffix will
    usually contain information about the model parameters, while the prefix
    defaults to 'model' if input value is None.
    """

    if file_prefix is None: file_prefix = 'model'
    model_dir = _check_dir_string(model_dir)
    filename = f"{model_dir}"
    try:
        model, history = load(
                                load_dir=model_dir,
                                file_prefix=file_prefix,
                                file_suffix=file_suffix
                            )

        return model, history

    except:
        sys.exit(
            f"Not able to load model from '{filename}' in '{model_dir}'.\n"\
            "Please verify that the file exists and is stored as a "\
            "list containing a 'model' and 'history' object."
        )

def load_from_list_of_models(model_dir='src/datastore/models/'):
    """Description."""

    model_dir = _check_dir_string(model_dir)
    models = os.listdir(model_dir)
    file_select = {}
    print(f"\n\nModels in '{model_dir}':'")
    for n in range(1, models.__len__() + 1):
        model = os.path.splitext(models[n-1])[0] # current model without ext
        file_select[n] = model
        print(f"{n}: {model}")

    print('-'*max(models).__len__())
    selector = input('Choose model number to load: ')
    try:
        selector = int(selector)
    except:
        sys.exit(
                'Invalid input type, the program has been terminated.\n'\
                'Please select a digit corresponding with desired model file.'
            )
    if int(selector) not in file_select:
        sys.exit('Invalid model number. The program has been terminated.')

    filename = file_select[selector]
    model, history = load(load_dir=model_dir,file_prefix=filename)
    return model, history

def _check_dir_string(dir_string):
    """Ensure that a directory location is given as an appendable string."""
    if dir_string[-1] != '/': dir_string=dir_string + '/'
    return dir_string

def _get_filename(prefix, suffix=None):
    """Returns a filename string based on a prefix and suffix property. If
    suffix is not declared, only the prefix value is used."""

    if suffix is not None: return f"{prefix}_{suffix}"
    else: return prefix

if __name__ == '__main__':
    import sys
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')