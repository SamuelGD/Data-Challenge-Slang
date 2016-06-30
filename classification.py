# -*-coding:Latin-1 -*
import numpy as np

class Classifier:
    
    def fit(self, X):
        n_samples, n_features = X.shape

        # Lagrange multipliers
        #TODO
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = sp.sparse.csc_matrix.dot(X[i], X[j].T).toarray()
        
        P = sp.sparse.csc_matrix(np.outer(y,y) * K)
        q = sp.sparse.csc_matrix(np.ones(n_samples) * -1)
        A = sp.sparse.csc_matrix(y)
        b = sp.sparse.csc_matrix(0.0)
        
        def fun_to_minimize(x):
            #if(x.shape==(1, n_samples)):
                #x = np.asarray(x).reshape(-1)
            #a = P.dot(x)
            #a = np.asarray(a).reshape(-1)
            z = (1/2)*np.dot(x,P.dot(x)) + q.dot(x)
            return(z)
        
        def fun_cons(x):
            z = A.dot(x)
            return(z)
        
        cons = ({'type': 'eq', 'fun': fun_cons})
        bnds = ((0, None),)*n_samples
        
        res = minimize(fun_to_minimize, (0,)*n_samples, method='SLSQP', bounds=bnds, constraints=cons)
        self.res = res
        a = res.x

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.ind = ind
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]
     
    def predict(self, X):
        self.w = np.asarray(self.w).reshape(-1)
        project = X.dot(self.w) + self.b
        return np.sign(project)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        
        return np.mean(y_pred == y)