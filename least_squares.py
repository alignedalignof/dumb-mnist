import numpy as np
import minst

class LeastSquares:
    def __init__(self, weights):
        self.weights = weights
        
    def classify(self, images):
        N = len(images) // minst.PIXEL_COUNT
        def label(images, ofs):
            M = np.array(images[ofs:ofs + minst.PIXEL_COUNT]).reshape((1, minst.PIXEL_COUNT))
            gaps = [np.dot(M, w) for w in self.weights]
            return np.argmax(gaps)
        return [label(images, minst.PIXEL_COUNT*i) for i in range(N)]
    
def train(images, labels):
    GAP = 1
    
    N = len(labels)
    M = np.array(images).reshape((N, minst.PIXEL_COUNT))
    U, S, Vh = np.linalg.svd(M, full_matrices = False)
    E = np.diag(S)
    W = np.dot(Vh.transpose(), np.linalg.inv(E))
    W = np.dot(W, U.transpose())
    Y = [np.array([(GAP if l == label else -GAP) for l in labels]).reshape((N, 1)) for label in range(minst.LABEL_COUNT)]
    return LeastSquares([np.dot(W, y) for y in Y])

