import numpy as np
import sys
import time
from random import randint
import argparse




def readData(file, nfeat):
	nline = len(open(file, 'r').readlines())
	X = np.zeros([nline, nfeat + 1])
	y = np.zeros(nline)
	i = 0
	for line in open(file, 'r').readlines():
		arr = line.strip().split()
		y[i] = int(arr[0])
		X[i][0] = 1
		for feat in arr[1:]:
			X[i][int(feat.split(':')[0])] = 1
		i += 1
	return (X, y)



def train(X, y, max_it):
    w = np.zeros(X.shape[1])
    N = y.shape[0]

    it = 0
    acc = 0.0

    while it < max_it:

        n = randint(0, N-1)

        if np.dot(w, X[n]) * y[n] <= 0:
            it += 1

            w += 1/it * y[n] * X[n]

            acc = predict(w, X, y)
            print('training...%d/%d\taccuracy=%f' % (it, max_it, acc), end='\r')

    return w






def predict(w, X, y):
    t = np.dot(X, w)
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
    args = parser.parse_args()

    max_it = int(args.max_it)



    # read data
    X_t, y_t = readData('a7a.train', 123)
    X_p, y_p = readData('a7a.test', 123)



    # train
    w = train(X_t, y_t, max_it)

    # predict
    print('\ntest set accuracy = %f' % predict(w, X_p, y_p))


    end_time = time.time()
    print('%f seconds' % (end_time - start_time))







