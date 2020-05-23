import pickle
import os
from datetime import datetime

def store(obj, filename='store'):
    f = open('src/datastore/{}.pckl'.format(filename), 'wb')
    pickle.dump(obj, f)
    f.close()
    print("Object succesfully stored as '{}.pckl' in src/datastore".format(filename))
    return

def store_time_interval(start, end, filename='store_timeint'):
    f = open('src/datastore/{}.pckl'.format(filename), 'wb')
    obj = []
    for timestamp in [start, end]:
        date_string = str(timestamp.day) + '-' + str(timestamp.month) + '-'+ str(timestamp.year)
        obj.append(date_string)
    pickle.dump(obj, f)
    f.close()
    print("Object succesfully stored as '{}.pckl' in src/datastore".format(filename))
    return

def load(filename='store'):
    f = open('src/datastore/{}.pckl'.format(filename), 'rb')
    obj = pickle.load(f)
    f.close()
    print("Object succesfully loaded from '{}.pckl' in src/datastore".format(filename))
    return obj

def load_meta(filename='store_meta'):
    return load(filename=filename)

def delete(filename='store'):
    path = 'src/datastore/{}.pckl'.format(filename)
    print(path)
    try:
        os.remove(path)
        print("Object stored in file '{}.pckl' deleted from src/datastore".format(filename))
    except:
        print("File '{}.pckl' not found in src/datastore.".format(filename))
    return

def save_model(model,history,modelstring='unspecificed',**kwargs):
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

    Add a time to the end of the file to give it a unique name
    """
    # for arg in kwargs:
    #     spec_string = arg + '-'
    # spec_string = spec_string[:-1]
    # modelstring = f"keras_model_{spec_string}_{datetime.time}"
    # f = open(modelstring, 'wb')
    json_model = model.to_json()
    f = open(f"src/datastore/models/model_{datetime.now().strftime('%Y%m%d-%H%M')}_{modelstring}.pckl", 'wb')
    pickle.dump([json_model, history.history], f)
    f.close()
    return

if __name__ == '__main__':
    import sys
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')