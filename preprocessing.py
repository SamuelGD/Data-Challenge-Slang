# -*-coding:Latin-1 -*
import numpy as np
import pandas as pd

""" Data preprocessing """

def bad_words(words, wordlist_fname):
    """ Returns the number of bad_words and ratio of bad_words """

    badwords = pd.read_csv("badwords.txt", sep = '\n').as_matrix()
    badwords = np.ndarray.flatten(badwords)
    
    mask = np.in1d(words, badwords)
    
    count = len(mask == True)
    ratio = float(count) / len(words)
    
    return count, ratio
    
def capital_words(words):
    """ Returns the number of uppercase words """
    
    pass
    

def preprocessing(X):
    """ Create features """

        
    
    X_processed = np.array([len(x) for x in X])
    
    
    
    return X_processed