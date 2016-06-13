# -*-coding:Latin-1 -*
import numpy as np
from preprocessing import preprocessing

""" Detection of insults in comments """

def load_data(train_fname, test_fname):
    """ Load the train data and test data """

    X = []
    y = []
    
    with open(train_fname) as f:
        for line in f:
            y.append(int(line[0]))
            X.append(line[5:-6])
            
    y = np.array(y)
    
    X_test = []
    with open(test_fname) as f:
        for line in f:
            X_test.append(line[3:-6])
    
    return X, y, X_test


if __name__ == "__main__":
    
    X, y, X_test = load_data("train.csv", "test.csv")
    
    print("Length of train data: %d " % len(X))
    print("Length of test data: %d" % len(X_test))
    
    X_processed = preprocessing(X)
    X_test_processed = preprocessing(X_test)