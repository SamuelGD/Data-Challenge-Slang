# -*-coding:Latin-1 -*
import numpy as np
from preprocessing import preprocessing
from classification import Classifier

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
    
    # Load the data
    
    X, y, X_test = load_data("train.csv", "test.csv")
    
    # Preprocessing
    
    X_processed = preprocessing(X)
    X_test_processed = preprocessing(X_test)
    
    # Classification
    
    clf = Classifier()
    clf.fit(X_processed)
    y_pred = clf.predict(X_test_processed)
    
    print("Score on train: %f" % clf.score(X_processed, y))
    
    # Save predictions to file
    
    np.savetxt('y_pred.txt', y_pred, fmt='%s')