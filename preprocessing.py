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
    
""" Delete noise such as \\n and replace some words like u --> you """
def clean(f):
    f = [x.lower() for x in f]
    f = [x.replace("\\n"," ") for x in f]        
    f = [x.replace("\\t"," ") for x in f]        
    f = [x.replace("\\xa0"," ") for x in f]
    f = [x.replace("\\xc2"," ") for x in f]

    #f = [x.replace(","," ").replace("."," ").replace(" ", "  ") for x in f]
    #f = [re.subn(" ([a-z]) ","\\1", x)[0] for x in f]  
    #f = [x.replace("  "," ") for x in f]

    f = [x.replace(" u "," you ") for x in f]
    f = [x.replace(" em "," them ") for x in f]
    f = [x.replace(" da "," the ") for x in f]
    f = [x.replace(" yo "," you ") for x in f]
    f = [x.replace(" ur "," you ") for x in f]
    #f = [x.replace(" ur "," your ") for x in f]
    #f = [x.replace(" ur "," you're ") for x in f]
    
    f = [x.replace("won't", "will not") for x in f]
    f = [x.replace("can't", "cannot") for x in f]
    f = [x.replace("i'm", "i am") for x in f]
    f = [x.replace(" im ", " i am ") for x in f]
    f = [x.replace("ain't", "is not") for x in f]
    f = [x.replace("'ll", " will") for x in f]
    f = [x.replace("'t", " not") for x in f]
    f = [x.replace("'ve", " have") for x in f]
    f = [x.replace("'s", " is") for x in f]
    f = [x.replace("'re", " are") for x in f]
    f = [x.replace("'d", " would") for x in f]
    
    f = [x.replace(".", " ") for x in f]
    
    return f

""" Create a list of each words of the sentence"""
def separate(X):
    X_separate = []
    for i in range(0,len(X)):
        X_separate.append(X[i].split())
        
    return X_separate

def preprocessing(X):
    """ Create features """

        
    
    X = clean(X)
    X_processed = separate(X)
    print(X_processed[0])
    
    
    
    return X_processed