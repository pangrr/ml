from __future__ import print_function
import numpy as np
import sys
import time
from random import randint
import argparse
import matplotlib
#matplotlib.use('Agg') #comment this line to plot in OS with GUI
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

    while it < n_iteration:
        n = randint(0, N-1) # stochastic training

        if np.dot(w, X[n]) * y[n] <= 0:
            it += 1

            w += 1.0/it * y[n] * X[n]

            acc = predict(w, X, y)

            acc_arr.append(acc)

            print('Training...%d/%d\tTrain accuracy %f' % (it, n_iteration, acc), end='\r')

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
    print()


    # read arguments
    parser = argparse.ArgumentParser(description='Perceptron Classifier')
    parser.add_argument('-i', '--iterations', help='number of iterations, default = 100')
    parser.add_argument('-t', '--test', type=str, help='assign a file of test data')
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


   ###################   test_data(data_file)      ####################
    if args.test:
        X_p, y_p = readData(args.test, 123)
        t = np.dot(X_p, w)

        print("\nPredict on %d new test cases:" % y_p.shape[0])

        for _t in t:
            if _t > 0:
                print(1)
            else:
                print(-1)

        print("The accuracy on the new test case is: {0:.0f}%\n".format(accuracy(y_p, t)*100))
   ##################   test_data(data_file)      #####################




   ###########  test on default data and show accuracy movement  ##########
    else:
        # predict
        print('\nTest accuracy\t%f' % predict(w, X_p, y_p))


        end_time = time.time()
        print('%f seconds\n' % (end_time - start_time))


        # plot figure of accuracy movement over iteration
        plt.plot(acc_arr)
        plt.savefig('perceptron_converge')
        plt.show()

   ###########  test default data and show accuracy movement  ##########



