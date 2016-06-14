# -*-coding:Latin-1 -*
import numpy as np

class Classifier:
    
    def fit(self, X):
        pass
    
    def predict(self, X):
        return np.empty((10, 1))
    
    def score(self, X, y):
        y_pred = self.predict(X)
        
        return np.mean(y_pred == y)