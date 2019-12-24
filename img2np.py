import numpy as np
import minst

class Img2Np:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def convert(self, label):
        image = [f for i, f in enumerate(self.images) if label == self.labels[i//minst.PIXEL_COUNT]]
        return np.array(image).reshape((len(image)//minst.PIXEL_COUNT, minst.PIXEL_COUNT))