import mnist
import multiprocessing as mp
import numpy as np

THREADS = 4

class MinDif:
    def __init__(self, M, labels):
        self.M = M
        self.labels = labels
    
    def label(self, ofs):
        M = np.array(self.images[ofs:ofs + mnist.PIXEL_COUNT]).reshape((1, mnist.PIXEL_COUNT))
        dif = self.M - M
        dif = dif*dif
        dif = np.sum(dif, axis = 1)
        return self.labels[np.argmin(dif)]
    
    def classify(self, images):
        N = len(images)//mnist.PIXEL_COUNT
        self.images = images
        procs = mp.Pool(processes = THREADS)
        return procs.map(self.label, [i*mnist.PIXEL_COUNT for i in range(N)])
    
def train(images, labels):
    N = len(images)//mnist.PIXEL_COUNT
    return MinDif(np.array(images).reshape(N, mnist.PIXEL_COUNT), labels)