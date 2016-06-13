# -*-coding:Latin-1 -*
import numpy as np

""" Data preprocessing """

def preprocessing(X):
    """ Create features """
    
    X_processed = np.array([len(x) for x in X])
    
    return X_processed