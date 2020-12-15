Dumb MNIST
======================
Classifier algorithms for MNIST like images written in python-numpy

Usage
-----
`python runner.py data/fashion`

Output
------
```
Unpacking data/fashion/train-images-idx3-ubyte.gz data/fashion/train-labels-idx1-ubyte.gz
Unpacking data/fashion/t10k-images-idx3-ubyte.gz data/fashion/t10k-labels-idx1-ubyte.gz
classifiers.least_squares training 60000 images...6.91 s, 8684.99 images/s; classifying 10000 images... 0.40 s, 25064.74 images/s
80.87% hit
0: 800/1000 (80.00%)
1: 954/1000 (95.40%)
2: 686/1000 (68.60%)
3: 853/1000 (85.30%)
4: 764/1000 (76.40%)
5: 779/1000 (77.90%)
6: 462/1000 (46.20%)
7: 918/1000 (91.80%)
8: 944/1000 (94.40%)
9: 927/1000 (92.70%)

classifiers.pca training 60000 images...38.67 s, 1551.60 images/s; classifying 10000 images... 1.28 s, 7798.42 images/s
42.79% hit
0: 276/1000 (27.60%)
1: 639/1000 (63.90%)
2: 356/1000 (35.60%)
3: 390/1000 (39.00%)
4: 612/1000 (61.20%)
5: 56/1000 (5.60%)
6: 231/1000 (23.10%)
7: 180/1000 (18.00%)
8: 716/1000 (71.60%)
9: 823/1000 (82.30%)

classifiers.nearnull training 60000 images...107.34 s, 558.97 images/s; classifying 10000 images... 0.31 s, 31815.84 images/s
46.26% hit
0: 435/1000 (43.50%)
1: 543/1000 (54.30%)
2: 371/1000 (37.10%)
3: 253/1000 (25.30%)
4: 517/1000 (51.70%)
5: 310/1000 (31.00%)
6: 159/1000 (15.90%)
7: 719/1000 (71.90%)
8: 621/1000 (62.10%)
9: 698/1000 (69.80%)

classifiers.mindif training 60000 images...1.37 s, 43886.52 images/s; classifying 100 images... 20.13 s, 4.97 images/s
85.00% hit
0: 8/8 (100.00%)
1: 13/13 (100.00%)
2: 12/14 (85.71%)
3: 7/9 (77.78%)
4: 6/10 (60.00%)
5: 8/9 (88.89%)
6: 5/8 (62.50%)
7: 9/11 (81.82%)
8: 12/12 (100.00%)
9: 5/6 (83.33%)

classifiers.mindif training 1000 images...0.02 s, 42645.84 images/s; classifying 10000 images... 28.93 s, 345.66 images/s
75.06% hit
0: 775/1000 (77.50%)
1: 932/1000 (93.20%)
2: 604/1000 (60.40%)
3: 699/1000 (69.90%)
4: 610/1000 (61.00%)
5: 624/1000 (62.40%)
6: 495/1000 (49.50%)
7: 920/1000 (92.00%)
8: 906/1000 (90.60%)
9: 941/1000 (94.10%)

classifiers.nn training 10000 images...45.76 s, 218.54 images/s; classifying 10000 images... 1.04 s, 9634.82 images/s
81.38% hit
0: 761/1000 (76.10%)
1: 961/1000 (96.10%)
2: 708/1000 (70.80%)
3: 874/1000 (87.40%)
4: 665/1000 (66.50%)
5: 896/1000 (89.60%)
6: 482/1000 (48.20%)
7: 914/1000 (91.40%)
8: 952/1000 (95.20%)
9: 925/1000 (92.50%)

classifiers.relu training 60000 images...344.75 s, 174.04 images/s; classifying 10000 images... 0.30 s, 33673.99 images/s
49.48% hit
0: 11/1000 (1.10%)
1: 970/1000 (97.00%)
2: 485/1000 (48.50%)
3: 221/1000 (22.10%)
4: 0/1000 (0.00%)
5: 326/1000 (32.60%)
6: 121/1000 (12.10%)
7: 911/1000 (91.10%)
8: 951/1000 (95.10%)
9: 952/1000 (95.20%)
```
