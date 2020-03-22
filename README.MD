Dumb MNIST
======================
Classifier algorithms for MNIST like images wirtten in python-numpy

Usage
-----
`python runner.py data/fashion`

Output
------
```
Unpacking data/fashion/train-images-idx3-ubyte.gz data/fashion/train-labels-idx1-ubyte.gz
Unpacking data/fashion/t10k-images-idx3-ubyte.gz data/fashion/t10k-labels-idx1-ubyte.gz
classifiers.least_squares training 60000 images...7.05 s, 8510.89 images/s; classifying 10000 images... 0.40 s, 25168.76 images/s
80.96% hit
0: 800/1000 (80.00%)
1: 955/1001 (95.40%)
2: 688/1002 (68.66%)
3: 856/1003 (85.34%)
4: 768/1004 (76.49%)
5: 784/1005 (78.01%)
6: 468/1006 (46.52%)
7: 925/1007 (91.86%)
8: 952/1008 (94.44%)
9: 936/1009 (92.77%)

classifiers.pca training 60000 images...43.46 s, 1380.58 images/s; classifying 10000 images... 1.29 s, 7747.10 images/s
43.05% hit
0: 276/1000 (27.60%)
1: 640/1001 (63.94%)
2: 358/1002 (35.73%)
3: 393/1003 (39.18%)
4: 616/1004 (61.35%)
5: 61/1005 (6.07%)
6: 237/1006 (23.56%)
7: 187/1007 (18.57%)
8: 724/1008 (71.83%)
9: 832/1009 (82.46%)

classifiers.nearnull training 60000 images...107.87 s, 556.21 images/s; classifying 10000 images... 0.31 s, 32377.00 images/s
46.50% hit
0: 435/1000 (43.50%)
1: 544/1001 (54.35%)
2: 373/1002 (37.23%)
3: 256/1003 (25.52%)
4: 521/1004 (51.89%)
5: 315/1005 (31.34%)
6: 165/1006 (16.40%)
7: 726/1007 (72.10%)
8: 629/1008 (62.40%)
9: 707/1009 (70.07%)

classifiers.mindif training 60000 images...1.32 s, 45307.97 images/s; classifying 100 images... 20.98 s, 4.77 images/s
89.66% hit
0: 8/8 (100.00%)
1: 14/14 (100.00%)
2: 14/16 (87.50%)
3: 10/12 (83.33%)
4: 10/14 (71.43%)
5: 13/14 (92.86%)
6: 11/14 (78.57%)
7: 16/18 (88.89%)
8: 20/20 (100.00%)
9: 14/15 (93.33%)

classifiers.mindif training 1000 images...0.03 s, 39992.60 images/s; classifying 10000 images... 29.55 s, 338.41 images/s
75.17% hit
0: 775/1000 (77.50%)
1: 933/1001 (93.21%)
2: 606/1002 (60.48%)
3: 702/1003 (69.99%)
4: 614/1004 (61.16%)
5: 629/1005 (62.59%)
6: 501/1006 (49.80%)
7: 927/1007 (92.06%)
8: 914/1008 (90.67%)
9: 950/1009 (94.15%)

classifiers.nn training 10000 images...45.48 s, 219.87 images/s; classifying 10000 images... 1.02 s, 9814.64 images/s
81.43% hit
0: 736/1000 (73.60%)
1: 952/1001 (95.10%)
2: 795/1002 (79.34%)
3: 797/1003 (79.46%)
4: 727/1004 (72.41%)
5: 912/1005 (90.75%)
6: 490/1006 (48.71%)
7: 898/1007 (89.18%)
8: 909/1008 (90.18%)
9: 964/1009 (95.54%)
```
