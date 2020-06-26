#-----------------------------------------------------------------------------
# File: model.py
# Purpose:
#   Template for creating and implementing machine learning models for
#   predictive maintenance using TensorFlow and Keras.
#
# Created by: Johan Fredrik Alvsaker
# Last modified: 03.06.2020
#-----------------------------------------------------------------------------
# Standard library:
import os, pickle, sys

# External modules:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

# Local API:
from src.api import file_management as filemag
from src.api import memory as mem
from src.api import modeling_funcs as mfnc
#-----------------------------------------------------------------------------
def create():
    """Implement your model creation function here."""
    sys.exit('Not implemented.')

    # return model

def train():
    """Implement your model training function here."""
    sys.exit('Not implemented.')

    # return model, history

def test():
    """Implement your model testing function here."""
    sys.exit('Not implemented.')

    # return performance

def visualize():
    """Implement your result visualization function here."""
    sys.exit('Not implemented.')

    # return

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(f'Run from manage.py, not {os.path.basename(__file__)}.')
