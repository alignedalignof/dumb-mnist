import sys
import minst
import least_squares
import mindif
import pca
import nearnull
import time
import gzip

def get_file_data(img, lbl):
    next_u32 = lambda file: int.from_bytes(file.read(4), "big")
    
    print("Unpacking", img, lbl)
    with gzip.open(img, "rb") as images, gzip.open(lbl, "rb") as labels:
        if next_u32(images) != minst.IMAGE_MAGIC:
            raise ValueError(img, "invalid magic")
        img_count = next_u32(images)
        h = next_u32(images)
        w = next_u32(images)
        if w != h or w != minst.WIDTH:
            raise ValueError("Invalid image dimensions", w, h)
        if next_u32(labels) != minst.LABEL_MAGIC:
            raise ValueError(lbl, "invalid magic")
        if img_count != next_u32(labels):
            raise ValueError(lbl, "invalid label count")
        return [b/255. for b in images.read(img_count*w*h)], [int(l) for l in labels.read(img_count)]

class Timer:
    def __init__(self):
        self.start = None
        self.dt = None
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dt = time.time() - self.start
    
def evaluate(module, images, labels, test_images, test_labels):
    print(module.__name__, "training", len(labels), "images...", end = "")
    with Timer() as train_timer:
        classifier = module.train(images, labels)
    print("{0:.2f} s, {1:.2f} images/s".format(train_timer.dt, len(labels)/train_timer.dt), end = "")
    
    print("; classifying", len(test_labels), "images...", end = "")
    with Timer() as class_timer:
        evaluation = classifier.classify(test_images)
    print(" {0:.2f} s, {1:.2f} images/s".format(class_timer.dt, len(test_labels)/class_timer.dt))
    if len(test_labels) != len(evaluation):
        raise ValueError("Invalid evaluation size")
    
    totals = [i for i in range(minst.LABEL_COUNT)]
    hits = [i for i in range(minst.LABEL_COUNT)]
    for e, a in zip(test_labels, evaluation):
        totals[e] += 1
        if e == a:
            hits[e] += 1

    print("{0:.2f}% hit".format(100.*sum(hits)/sum(totals)))
    for t, h, i in zip(totals, hits, range(minst.LABEL_COUNT)):
        print("{3}: {0}/{1} ({2:.2f}%)".format(h, t, 100.*h/t, i))
    print()
    
if __name__ == "__main__":
    folder = "data/mnist"
    if len(sys.argv) == 2:
        folder = sys.argv[1]
    elif len(sys.argv) > 2:
        raise ValueError("Invalid arguments(count)", sys.argv)
    
    train_img = folder + "/train-images-idx3-ubyte.gz"
    train_lbl = folder + "/train-labels-idx1-ubyte.gz"
    test_img =  folder + "/t10k-images-idx3-ubyte.gz"
    test_lbl =  folder + "/t10k-labels-idx1-ubyte.gz"
    
    images, labels = get_file_data(train_img, train_lbl)
    test_images, test_labels = get_file_data(test_img, test_lbl)
    
    evaluate(least_squares, images, labels, test_images, test_labels)
    evaluate(pca, images, labels, test_images, test_labels)
    evaluate(nearnull, images, labels, test_images, test_labels)
            
    N = 100
    evaluate(mindif, images, labels, test_images[:N*minst.PIXEL_COUNT], test_labels[:N])
    N = 1000
    evaluate(mindif, images[:N*minst.PIXEL_COUNT], labels[:N], test_images, test_labels)