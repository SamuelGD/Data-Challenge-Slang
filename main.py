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
    print("Loading data")
    
    X, y, X_real_test = load_data("train.csv", "test.csv")
    
    
    # Preprocessing
    
    print("Beginning preprocessing")
    
    X_processed, X_test_processed = preprocessing(X, X_real_test)

    assert X_processed.shape[1] == X_test_processed.shape[1] # check that train and test have the same number of features
    
    # Classification
    
    print("Beginning prediction")
    
    #clf = Classifier()
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42)
    svc = SVC(C = 50000) # best_score: 0.828 pour C = 50000 (4 gram et tfidf)
    
    Cs = [50000]
    clf = GridSearchCV(estimator = svc, param_grid = dict(C = Cs), n_jobs = -1)
    #clf = svc    
    #X_train = X_processed
    #y_train = y
    clf.fit(X_train, y_train)
    
    print(clf.best_estimator_)
    
    print("Score: %f" % clf.score(X_test, y_test))
    
    
    print("Saving prediction to file")
    
    y_pred = clf.predict(X_test_processed)
    
    # Save predictions to file
    
    np.savetxt('y_pred.txt', y_pred, fmt='%s')
