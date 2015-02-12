import numpy as np
import sys
import time
from random import randint
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Process, Array, Value





def readData(file, nfeat):
	nline = len(open(file, 'r').readlines())
	X = np.zeros([nline, nfeat])
	y = np.zeros(nline)
	i = 0
	for line in open(file, 'r').readlines():
		arr = line.strip().split()
		y[i] = int(arr[0])
		for feat in arr[1:]:
			X[i][int(feat.split(':')[0])-1] = 1
		i += 1
	return (X, y)



def train(X, y, max_it, C):
    w = np.zeros(X.shape[1])
    b = 0

    N = y.shape[0]

    it = 0
    acc = 0.0

    while it < max_it:
        n = randint(0, N-1)


        if y[n] * (np.dot(w, X[n]) + b) < 1:
            it += 1

            w -= 1/it * (w - C * y[n] * X[n])
            b += 1/it * C * y[n] #???

            acc = predict(w, b, X, y)
            print('training...%d/%d\taccuracy=%f' % (it, max_it, acc), end='\r')
        else:
            w -= 1/it * w

    return (w, b)






def predict(w, b, X, y):
    t = np.dot(np.insert(X, X.shape[1], values=1, axis=1), np.insert(w, w.shape[0], values=b))
    return accuracy(y, t)




def accuracy(y, t):
	match = 0.0
	for i in range(y.shape[0]):
		if y[i] * t[i] > 0:
			match += 1
	return match / y.shape[0]








if __name__ == '__main__':

    start_time = time.time()


    # read arguments
    parser = argparse.ArgumentParser(description='Perceptron Classifier')
    parser.add_argument('-i', '--max_it', help='Maximum number of iterations', required=True)
    parser.add_argument('-c', '--c', help='C', required=True)
    args = parser.parse_args()

    max_it = int(args.max_it)
    C = float(args.c)



    # read data
    X_t, y_t = readData('a7a.train', 123)
    X_p, y_p = readData('a7a.test', 123)



    # train
    w, b  = train(X_t, y_t, max_it, C)

    # predict
    print('\ntest set accuracy = %f' % predict(w, b, X_p, y_p))


    end_time = time.time()
    print('%f seconds' % (end_time - start_time))







