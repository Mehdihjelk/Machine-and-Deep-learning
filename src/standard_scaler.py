import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.scale= None

    def fit_transform(self,X):
        self.mean=np.mean(X, axis=0)
        self.scale=np.std(X,axis=0)

        self.scale=np.where(self.scale==0, 1e-9, self.scale)

        return (X-self.mean)/ self.scale
    
    #for new data
    def transform(self,X):
        return (X-self.mean)/ self.scale