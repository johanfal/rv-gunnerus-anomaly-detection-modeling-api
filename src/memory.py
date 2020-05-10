import pickle
import os

def store(obj, filename='store'):
    f = open('src/datastore/{}.pckl'.format(filename), 'wb')
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

def loadMeta(filename='store_meta'):
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