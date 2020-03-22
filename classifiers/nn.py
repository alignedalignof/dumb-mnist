import mnist
import numpy as np
import multiprocessing as mp
import img2np

def sig(z):
    return 1/(1 + np.exp(-z));

class NN:
    def __init__(self, arch):
        if (len(arch) < 2):
            raise ValueError("1-layer NN")
        self.W = [np.random.randn(m, n + 1) for n, m in zip(arch[:-1], arch[1:])] #does most of the work
    
    def ignite(self, x):
        self.Y = [np.append(x, np.array([[1]]), axis = 0)]
        self.Z = []
        for w in self.W:
            self.Z.append(np.dot(w, self.Y[-1]))
            y = sig(self.Z[-1])
            self.Y.append(np.append(y, np.array([[1]]), axis = 0))
            
    def fire(self, x):
        self.ignite(x)
        O = self.Y[-1]
        return O[:-1,:];
    
    def batch(self, group):
        Ms = [m.reshape((mnist.PIXEL_COUNT, 1)) for m in group]
        dW = [np.zeros(w.shape) for w in self.W]
        for i, m in enumerate(Ms):
            O = self.fire(m)
            T = np.zeros((mnist.LABEL_COUNT, 1))
            T[i, 0] = 1.0
            E = O - T
            dZ = [E*sig(self.Z[-1])*(1 - sig(self.Z[-1]))]
            for z, w in zip(self.Z[-2::-1], self.W[-1:0:-1]):
                dY = np.dot(w.T, dZ[0]) #hax
                dY = dY[:-1, :]
                dZ.insert(0, dY*sig(z)*(1 - sig(z)))
            for y, dz, dw, in zip(self.Y[:-1], dZ, dW):
                dw += np.dot(dz, y.T)
        for w, dw in zip(self.W, dW):
            np.subtract(w, 0.5*dw, out = w) #hax
        
    def train(self, images, labels):
        procs = mp.Pool(processes = 4)
        Ms = procs.map(img2np.Img2Np(images, labels).convert, range(mnist.LABEL_COUNT))
        for i in range(30): #hax
            for group in zip(*Ms):
                self.batch(group)
                
    def classify(self, images):
        N = len(images)//mnist.PIXEL_COUNT
        def label(ofs):
            M = np.array(images[ofs:ofs + mnist.PIXEL_COUNT]).reshape((mnist.PIXEL_COUNT, 1))
            M = self.fire(M)
            return np.argmax(M, axis = 0)
        return [label(i*mnist.PIXEL_COUNT) for i in range(N)]

def train(images, labels):
    nn = NN([mnist.PIXEL_COUNT, mnist.WIDTH, mnist.LABEL_COUNT])
    nn.train(images, labels)
    return nn