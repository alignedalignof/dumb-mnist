import mnist
import numpy as np
import multiprocessing as mp
import img2np

THREADS = 4

class Pca:
    def __init__(self, means, projs):
        self.means = means
        self.projs = projs
        
    def classify(self, images):
        N = len(images) // mnist.PIXEL_COUNT
        def label(images, ofs):
            M = np.array(images[ofs:ofs + mnist.PIXEL_COUNT]).reshape((mnist.PIXEL_COUNT, 1))
            farthest = 0
            label = mnist.LABEL_COUNT
            for lbl, mean, proj in zip(range(mnist.LABEL_COUNT), self.means, self.projs):
                o = np.dot(proj, M)
                d = np.linalg.norm(o - mean)
                if d > farthest:
                    farthest = d
                    label = lbl
            return label
        return [label(images, i*mnist.PIXEL_COUNT) for i in range(N)]
    
def train(images, labels):
    procs = mp.Pool(processes = THREADS)
    Ms = [M.T for M in procs.map(img2np.Img2Np(images, labels).convert, range(mnist.LABEL_COUNT))]
    def projector(label):
        C = np.dot(Ms[label], Ms[label].T)
        W, V = np.linalg.eigh(C)
        return V[:, -1:-3:-1].T
    projs = [projector(label) for label in range(mnist.LABEL_COUNT)]
    def mean(label):
        O = np.concatenate([o for l, o in enumerate(Ms) if l != label], axis = 1)
        O = np.dot(projs[label], O)
        return np.mean(O, axis = 1).reshape((2, 1))
    return Pca([mean(label) for label in range(mnist.LABEL_COUNT)], projs)