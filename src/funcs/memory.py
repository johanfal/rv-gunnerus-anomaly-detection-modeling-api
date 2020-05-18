import pickle
import os

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

if __name__ == '__main__':
    import sys
    sys.exit('Run from manage.py, not memory.')