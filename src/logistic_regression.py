import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.w=None
        self.b=None
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def logisticloss(self,y,y_pred):
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) #Tto avoid prob 0 or 1
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def fit(self, X,y):
        self.w=np.zeros(X.shape[1])
        self.b=0
        self.losses=[]
        #Gradient descent to find the wieghts w and b
        for i in range(self.n_iterations):
            y_pred=self.sigmoid(X@self.w+self.b)
            current_loss=self.logisticloss(y,y_pred)
            self.losses.append(current_loss)
            grad_w=1/X.shape[0]*(X.T@(y_pred-y))
            grad_b=1/X.shape[0]*np.sum(y_pred-y)
            self.w-=self.learning_rate*grad_w
            self.b-=self.learning_rate*grad_b

    def predict(self,X):
        y_pred=self.sigmoid(X@self.w+self.b)
        return (y_pred>=0.5).astype(int)