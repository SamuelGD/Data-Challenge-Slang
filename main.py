# -*-coding:Latin-1 -*
import numpy as np
from preprocessing import preprocessing
from classification import Classifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

""" Detection of insults in comments """

def load_data(train_fname, test_fname):
    """ Load the train data and test data """

    X = []
    y = []
    
    with open(train_fname) as f:
        for line in f:
            y.append(int(line[0]))
            X.append(line[5:-4])
            
    y = np.array(y)
    
    X_test = []
    with open(test_fname) as f:
        for line in f:
            X_test.append(line[3:-4])
    
    return X, y, X_test


if __name__ == "__main__":
    
    # Load the data
    
    X, y, X_real_test = load_data("train.csv", "test.csv")
    
    
    # Preprocessing
    
    X_processed, X_dbg = preprocessing(X)
    
    print(X_dbg[0])

    #X_test_processed = preprocessing(X_real_test)
    
    # Classification
    
    print("Beginning prediction")
    
    #clf = Classifier()
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42)
    svc = SVC(C = 1.0) # best_score: 0.815217 for C = 990000
    
    Cs = [990000, 900000, 1100000]
    clf = GridSearchCV(estimator = svc, param_grid = dict(C = Cs), n_jobs = -1)
    clf.fit(X_train, y_train)
    
    print(clf.best_estimator_)
    
    print("Score: %f" % clf.score(X_test, y_test))
    
    
    #y_pred = clf.predict(X_test_processed)
    
    # Save predictions to file
    
    #np.savetxt('y_pred.txt', y_pred, fmt='%s')
