# -*-coding:Latin-1 -*
import numpy as np
import pandas as pd
import re

""" Data preprocessing """

    
def bad_words(documents, wordlist_fname):
    """ Creates 2 features: count and ratio of bad words """
    
    
    def bad_words_text(document, wordlist_fname):
        """ Returns the number of bad_words and ratio of bad_words in a single example """
    
        badwords = pd.read_csv("badwords.txt", sep = '\n').as_matrix()
        badwords = np.ndarray.flatten(badwords)
        
        mask = np.in1d(document, badwords)
        
        count = len(mask[mask == True])
        ratio = float(count) / len(document)
        
        return count, ratio
    
    v_bad_words = np.vectorize(bad_words_text)
    
    count, ratio = v_bad_words(documents, wordlist_fname)
    
    count = count.reshape((-1, 1))
    ratio = ratio.reshape((-1, 1))
    
    X = np.hstack([count, ratio])
    
    return X
    
def uppercase_words(documents):
    """ Creates feature: ratio of uppercase words """
    
    def uppercase_words_text(document):
        """ Returns ratio of uppercase words in a single document """
        
        v_is_upper = np.vectorize(str.isupper)
        
        mask = v_is_upper(document)
        
        ratio = np.mean(mask == True)
        
        return ratio
            
    v_uppercase_words_text = np.vectorize(uppercase_words_text)
    
    X = v_uppercase_words_text(documents).reshape((-1, 1))
    
    return X

def exclamation_marks(documents):
    """ Creates one feature: ratio of exclamation marks """
    
    def exclamation_marks_text(document):
        """ Returns ratio of exclamation marks in a single document (compared to the number of words) """
        
        count = 0
        
        for word in document:
            count += word.count('!')
        
        ratio = float(count) / len(document)
        
        return ratio
    
    v_exclamation_marks_text = np.vectorize(exclamation_marks_text)
    
    X = v_exclamation_marks_text(documents).reshape((-1, 1))
    
    return X
    
def smileys(documents):
    """ Creates one feature: ratio of nice smileys """
    
    list_smileys = [':)', ':-)', ';)', ';-)', '=)', '=D', ':p', ':P', '<3']
    
    def smileys_text(document):
        """ Returns number of smileys in a single document """
        
        count = 0
        
        for word in document:
            for smiley in list_smileys:
                count += word.count(smiley)
        
        return count
        
    v_smileys_text = np.vectorize(smileys_text)
    
    X = v_smileys_text(documents).reshape((-1, 1))
    
    return X
    

def clean(f):
    """ Deletes noise such as \\n and replace some words like u --> you """
    
    f = [x.replace("\\n"," ") for x in f]        
    f = [x.replace("\\t"," ") for x in f]        
    f = [x.replace("\\xa0"," ") for x in f]
    f = [x.replace("\\xc2"," ") for x in f]
    f = [x.replace("\\xbf"," ") for x in f]
    f = [x.replace("\\r"," ") for x in f]
    f = [x.replace("\\"," ") for x in f]

    f = [x.replace(" u "," you ") for x in f]
    f = [x.replace(" em "," them ") for x in f]
    f = [x.replace(" da "," the ") for x in f]
    f = [x.replace(" yo "," you ") for x in f]
    f = [x.replace(" ur "," you ") for x in f]
    
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
    
    def remove_urls(document):
        """ Removes all urls in document. """
        
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', document)
        
        for url in urls:
            document = document.replace(url, " ")
            
        return document
    
    f = [remove_urls(x) for x in f]
    
    f = [x.replace(".", " ") for x in f]
    f = [x.replace("?"," ") for x in f]
    f = [x.replace("-"," ") for x in f]
    f = [x.replace("^"," ") for x in f]
    f = [x.replace("[", " ") for x in f]
    f = [x.replace("]", " ") for x in f]
    
    return f

def clean_twice(documents):
    """
    Deletes remaining noise such as '!' and ')'
    This function should be called when we are sure these symbols won't be usefull anymore
    """
    
    clean_documents = []
    
    for doc in documents:
        doc = [x.replace("!","") for x in doc]
        doc = [x.replace("(","") for x in doc]
        doc = [x.replace(")","") for x in doc]
        doc = [x.replace(":","") for x in doc]
        
        clean_documents.append(doc)
    
    return clean_documents

def separate(X):
    """ Create a list of each words of the sentence"""
    
    X_separate = []
    for i in range(0,len(X)):
        X_separate.append(X[i].split())
        
    return X_separate
    
def all_lowercase(documents):
    """ Make sure all letters are lowercase """
    
    lower_documents = []
    
    for doc in documents:

        lower_documents.append([x.lower() for x in doc])
        
    return lower_documents
    

def preprocessing(X):
    """ Create features """

    X = clean(X)
    X = separate(X)
    X = [x for x in X if x]
    
    X_bad_words = bad_words(X, "badwords.txt")
    X_uppercase = uppercase_words(X)
    X_exclamation_marks = exclamation_marks(X)
    X_smileys = smileys(X)
    
    X = clean_twice(X)
    X = all_lowercase(X)
    
    X_processed = np.hstack([X_bad_words, X_uppercase, X_exclamation_marks, X_smileys])
    
    return X_processed
