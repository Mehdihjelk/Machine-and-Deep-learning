import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.w=None
        self.b=None
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    @staticmethod
    def logisticloss(y,y_pred):
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

class LogisticRegressionL2(LogisticRegression):
    def __init__(self, learning_rate=0.01, n_iterations=100, lambda_=0.001):
        super().__init__(learning_rate, n_iterations)
        self.lambda_=lambda_

    def logisticloss(self,y,y_pred):
        basicloss=super().logisticloss(y,y_pred)
        penality_L2=self.lambda_ * np.sum(self.w**2)
        return basicloss + penality_L2
    
    def fit(self, X,y):
        self.w=np.zeros(X.shape[1])
        self.b=0
        self.losses=[]
        #Gradient descent to find the wieghts w and b
        for i in range(self.n_iterations):
            y_pred=self.sigmoid(X@self.w+self.b)
            current_loss=self.logisticloss(y,y_pred)
            self.losses.append(current_loss)
            grad_w=1/X.shape[0]*(X.T@(y_pred-y)) + 2*self.lambda_ * self.w
            grad_b=1/X.shape[0]*np.sum(y_pred-y)
            self.w-=self.learning_rate*grad_w
            self.b-=self.learning_rate*grad_b

class LogisticRegressionmulticlass:
    def __init__(self,learning_rate=0.01,n_iterations=100):
        self.w=None
        self.b=None
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def crossentropyloss(y,y_pred):
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -np.mean(np.sum(y*np.log(y_pred), axis=1))

    def fit(self,X,y):
        self.w=np.zeros((X.shape[1],len(np.unique(y))))
        self.b=np.zeros(len(np.unique(y)))
        self.losses=[]
        Y_one_hot = np.zeros((X.shape[0], len(np.unique(y)))) #to convert the labels to one hot encoding (matrix of 0 and 1)
        Y_one_hot[np.arange(X.shape[0]), y] = 1
        for i in range(self.n_iterations):
            z=X@self.w +self.b
            y_pred=self.softmax(z)

            currentloss=self.crossentropyloss(Y_one_hot,y_pred)
            self.losses.append(currentloss)
            grad_w=(1/X.shape[0]) * (X.T)@(y_pred - Y_one_hot)
            grad_b=(1/X.shape[0])* np.sum(y_pred-Y_one_hot,axis=0)
            self.w=self.w - self.learning_rate*grad_w
            self.b=self.b -self.learning_rate*grad_b
    
    def predict(self, X):
        z=X@self.w +self.b
        proba=self.softmax(z)
        prediction=np.argmax(proba, axis=1)
        return prediction