#-----------------------------------------------------------------------------
# File: memory.py
# Purpose:
#   
#
# Created by: Johan Fredrik Alvsaker
# Last modified: 
#-----------------------------------------------------------------------------
# Standard library:
import os, pickle, sys
from datetime import datetime

# External modules:
from tensorflow import keras
from tensorflow.keras.models import load_model, model_from_json
#-----------------------------------------------------------------------------

def store(
            obj,
            store_dir:str='src/datastore/',
            file_prefix:str=None,
            file_suffix:str=None
        ) -> None:
    """Description."""

    if file_prefix is None: file_prefix = 'store'
    store_dir = _check_dir_string(store_dir)
    filename = _get_filename(file_prefix, file_suffix)

    f = open(f'{store_dir}{filename}.pckl', 'wb')
    pickle.dump(obj, f)
    f.close()
    print(f"Object succesfully stored as '{filename}.pckl' in '{store_dir}'")
    return

def load(
            load_dir:str='src/datastore/',
            file_prefix:str=None,
            file_suffix:str=None,
            verbose:bool=True
    ):
    """Description."""
    if file_prefix is None: file_prefix = 'store'
    load_dir = _check_dir_string(load_dir)
    filename = _get_filename(file_prefix, file_suffix)

    try:
        f = open(f'{load_dir}{filename}.pckl', 'rb')
        obj = pickle.load(f)
        f.close()
        if verbose:
            print(f"Object succesfully loaded from '{filename}' in '{load_dir}'")
        return obj
    except:
        sys.exit(f"{filename}.pckl not found in '{load_dir}'.")

def delete(
            delete_dir:str='src/datastore/',
            file_prefix:str=None,
            file_suffix:str=None
        ) -> None:
    """Description."""

    if file_prefix is None: file_prefix = 'store'
    delete_dir = _check_dir_string(delete_dir)
    filename = _get_filename(file_prefix, file_suffix)
    path = f'{delete_dir}{filename}.pckl'
    print(path)
    try:
        os.remove(path)
        print(f"File '{filename}.pckl' deleted from '{delete_dir}'")
    except:
        print(f"File '{filename}.pckl' not found in '{delete_dir}'.")
    return

def save_model(
                model:'tensorflow.keras.model',
                history:dict,
                model_dir:str='src/datastore/models',
                file_prefix:str=None,
                modelstring:str='unspecificed'
            ) -> None:
    """Saves a Keras model to a file named after important properties and
    tuning parameters of the model. Each saved file receives a unique name
    based on the time of save. Thus, no existing models are overwritten."""

    if file_prefix is None: file_prefix = 'model'
    json_model = model.to_json()
    model_dir = _check_dir_string(model_dir)
    unique_time = datetime.now().strftime('%Y%m%d-%H%M')
    tag = f"{unique_time}_{modelstring}"
    model_filename = f"{file_prefix}_{tag}"
    history_filename = f"{file_prefix}_{unique_time}_{modelstring}"
    path = f"{model_dir}{file_prefix}_{tag}"
    model.save(f"{path}.h5")
    f = open(f"{model_dir}{file_prefix}_history_{tag}.pckl",'wb')
    pickle.dump(history, f)
    f.close()
    print(f"Model and history with tag '{tag}' succesfully stored in " \
            f"'{model_dir}'.")
    return

def load_from_list_of_models(model_dir:str='src/datastore/models/'
                        ) -> ['tensorflow.keras.model', list]:
    """Description."""

    model_dir = _check_dir_string(model_dir)
    models = os.listdir(model_dir)
    if models.__len__() > 2:
        histories = {}
        file_select = {}
        print(f"\n\nModels in '{model_dir}':'")
        for n in range(1, models.__len__() + 1):
            # Get current model without file extension:
            model = os.path.splitext(models[n-1])[0]
            if 'history' in model:
                name_split = model.split('_history_',1)
                histories['_'.join(name_split)] = model
                continue
            file_select[n] = model
            print(f"{n}: {model}")

        print('-'*(max(file_select.values()).__len__()+3))
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
        model_file = file_select[selector]
        history_file = histories[model_file]
    else:
        for model in models:
            model = os.path.splitext(model)[0]
            if 'history' in model:
                history_file = model
                name_split = model.split('_history_',1)
            else:
                model_file = model

    model = load_model(f"{model_dir}{model_file}.h5")
    history = load(
        load_dir=model_dir,
        file_prefix=history_file,
        verbose=False
    )
    print(f"Model and history with tag '{model_file}' succesfully " \
        f"loaded from '{model_dir}'.")
    return model, history

def _check_dir_string(dir_string:str) -> str:
    """Ensure that a directory location is given as an appendable string."""
    if dir_string[-1] != '/': dir_string=dir_string + '/'
    return dir_string

def _get_filename(prefix:str, suffix:str=None) -> str:
    """Returns a filename string based on a prefix and suffix property. If
    suffix is not declared, only the prefix value is used."""

    if suffix is not None: return f"{prefix}_{suffix}"
    else: return prefix

if __name__ == '__main__':
    import sys
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
