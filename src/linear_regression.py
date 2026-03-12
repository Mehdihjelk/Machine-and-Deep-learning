import numpy as np

class LinearRegression:
    def __init__(self):
        self.w=None
    
    def fit(self,X,Y):
        self.w=np.linalg.inv(X.T@X)@X.T@Y
        
    def predict(self,X):
        return X@self.w