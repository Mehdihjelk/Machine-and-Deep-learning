import numpy as np

class LinearRegression:
    def __init__(self):
        self.w=None
    
    def fit(self,X,Y):
        self.w=np.linalg.inv(X.T@X)@X.T@Y
    
    def fit_ridge(self,X,Y,alpha):
        self.w=np.linalg.inv(X.T@X+alpha*(X.shape[0])*np.eye(X.shape[1]))@X.T@Y
    
    def predict(self,X):
        return X@self.w