# -*-coding:Latin-1 -*
import numpy as np

class Classifier:
    
    def fit(self, X):
        pass
    
    def predict(self, X):
        y_pred = (X < 60).astype(np.int)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        
        return np.mean(y_pred == y)