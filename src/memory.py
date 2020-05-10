import pickle
import os

def store(obj, filename='store'):
    f = open('src/datastore/store.pckl', 'wb')
    pickle.dump(obj, f)
    f.close()
    print("Object succesfully stored as '{}.pckl' in src/datastore".format(filename))
    return

def load(filename='store'):
    f = open('src/datastore/store.pckl', 'rb')
    obj = pickle.load(f)
    f.close()
    print("Object succesfully loaded from '{}.pckl' in src/datastore".format(filename))
    return obj

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