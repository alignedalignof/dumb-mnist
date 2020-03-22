import mnist
import numpy as np
import multiprocessing as mp
import img2np

THREADS = 4

class NearNull:
    def __init__(self, nulls):
        self.nulls = np.concatenate(nulls, axis = 1)
        
    def classify(self, images):
        N = len(images)//mnist.PIXEL_COUNT
        def label(ofs):
            M = np.array(images[ofs:ofs + mnist.PIXEL_COUNT]).reshape((1, mnist.PIXEL_COUNT))
            M = np.dot(M, self.nulls)
            M = np.abs(M)
            return np.argmin(M, axis = 1)
        return [label(i*mnist.PIXEL_COUNT) for i in range(N)]

def train(images, labels):
    procs = mp.Pool(processes = THREADS)
    Ms = procs.map(img2np.Img2Np(images, labels).convert, range(mnist.LABEL_COUNT))
    def null(label):
        O = np.concatenate([o for l, o in enumerate(Ms) if l != label], axis = 0)
        M = Ms[label]
        U, S, Vh = np.linalg.svd(M, full_matrices = True)
        A = np.dot(M, Vh.T)
        B = np.dot(O, Vh.T)
        A = np.count_nonzero(np.abs(A) < 1e-3, axis = 0)
        B = np.count_nonzero(np.abs(B) > 1e-3, axis = 0)
        last = np.argmax(B/(mnist.LABEL_COUNT - 1) + A)
        return Vh.T[:, last].reshape((mnist.PIXEL_COUNT, 1))
    return NearNull([null(label) for label in range(mnist.LABEL_COUNT)])