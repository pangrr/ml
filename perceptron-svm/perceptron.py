from __future__ import print_function
import numpy as np
import sys
import time
from random import randint
import argparse
import matplotlib
#plot without X window
#matplotlib.use('Agg')
import matplotlib.pyplot as plt




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



def train(X, y):
    global n_iteration
    global acc_arr

    w = np.zeros(X.shape[1])
    N = y.shape[0]

    it = 0
    acc = 0.0
    max_acc = 0.0

    while it < n_iteration:

        n = randint(0, N-1)

        if np.dot(w, X[n]) * y[n] <= 0:
            it += 1

            w += 1/it * y[n] * X[n]

            acc = predict(w, X, y)

            acc_arr.append(acc)

            print('Training...%d/%d\tTrain accuracy %f' % (it, n_iteration, acc), end='\r')
            if acc > max_acc:
                max_acc = acc
                best_w = w

    print('\nMax train accuracy %f' % (max_acc))
    return best_w






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
    parser.add_argument('-i', '--iterations', help='Number of iterations, default = 100')
    args = parser.parse_args()

    if args.iterations:
        n_iteration = int(args.iterations)
    else:
        n_iteration = 100



    # read data
    X_t, y_t = readData('a7a.train', 123)
    X_p, y_p = readData('a7a.test', 123)


    acc_arr = []

    # train
    w = train(X_t, y_t)

    # predict
    print('Test accuracy\t%f' % predict(w, X_p, y_p))


    end_time = time.time()
    print('%f seconds' % (end_time - start_time))

    plt.plot(acc_arr)
    plt.savefig('accuracy')
    plt.show()





