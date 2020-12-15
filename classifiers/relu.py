import mnist
import numpy as np

def sig(z):
    return 1/(1 + np.exp(-z));

class ReluNet:
    def __init__(self, arch):
        if (len(arch) < 2):
            raise ValueError("1-layer NN")
        self.W = [np.zeros((m, n)) for m, n in zip(arch[1:], arch[:-1])]
        for W in self.W:
            np.fill_diagonal(W, 0.42)
        self.B = [np.zeros((W.shape[0], 1)) for W in self.W]
    
    def evaluate(self, X):
        self.A = [X]
        for W, B in zip(self.W[:-1], self.B[:-1]):
            self.A.append(np.maximum(np.dot(W, self.A[-1]) + B, 0))
        self.A.append(sig(np.dot(self.W[-1], self.A[-1]) + self.B[-1]))
        return self.A[-1]
    
    def fit(self, X, Y):
        _, M = X.shape
        self.evaluate(X)
        dZ = self.A[-1] - Y
        for W, B, A in zip(self.W[-1::-1], self.B[-1::-1], self.A[-2::-1]):
            B -= np.mean(dZ, axis = 1, keepdims=True)*0.42
            dZin = dZ
            dZ = np.dot(W.T, dZin)*(A > 0)
            W -= 0.42/M*np.dot(dZin, A.T)
        
    def train(self, images, labels):
        M = len(images)//mnist.PIXEL_COUNT
        X = np.array(images).reshape((M, mnist.PIXEL_COUNT)).T
        Y = np.eye(mnist.LABEL_COUNT)[:,labels]
        for _ in range(500):
            self.fit(X, Y)
                
    def classify(self, images):
        M = len(images)//mnist.PIXEL_COUNT
        X = np.array(images).reshape((M, mnist.PIXEL_COUNT)).T
        Y = self.evaluate(X)
        return np.argmax(Y, axis = 0).tolist()

def train(images, labels):
    nn = ReluNet([mnist.PIXEL_COUNT, mnist.WIDTH, mnist.WIDTH, mnist.LABEL_COUNT])
    nn.train(images, labels)
    return nn