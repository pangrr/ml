from __future__ import print_function
import numpy as np
import sys
import time
from random import randint
import argparse
import matplotlib
#plot without X window
matplotlib.use('Agg')
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



def train(X, y, c):
    global n_iteration
    global acc_arr_single_c
    global args

    w = np.zeros(X.shape[1])
    b = 0

    N = y.shape[0]

    it = 0
    acc = 0.0
    max_acc = 0.0

    while it < n_iteration:
        n = randint(0, N-1)

        if y[n] * (np.dot(w, X[n]) + b) < 1:
            it += 1

            w -= 1.0/it * (w - c * y[n] * X[n])
            b += 1.0/it * c * y[n]

            acc = predict(w, b, X, y)

            if args.c:  # single c
                acc_arr_single_c.append(acc)
                print('Training...%d/%d\tTrain accuracy %f' % (it, n_iteration, acc), end='\r')

            # update max acc
            if acc > max_acc:
                max_acc = acc
                best_w = w
                best_b = b


        else:
            w -= 1.0/it * w

    if args.c:
        print('\nMax train accuracy\t%f' % (max_acc))
    return (best_w, best_b)






def predict(w, b, X, y):
    t = np.dot(np.insert(X, X.shape[1], values=1, axis=1), np.insert(w, w.shape[0], values=b))
    return accuracy(y, t)




def accuracy(y, t):
	match = 0.0
	for i in range(y.shape[0]):
		if y[i] * t[i] > 0:
			match += 1
	return match / y.shape[0]






def testC(pid, c_arr, acc_arr, X_t, y_t, X_d, y_d, done):
    global n_c
    global n_process

    i = pid
    while i < len(c_arr):
        w, b = train(X_t, y_t, c_arr[i])
        acc_arr[i] = predict(w, b, X_d, y_d)

        done.value += 1
        print('testing C...%d/%d' % (done.value, n_c), end='\r')

        i += n_process




if __name__ == '__main__':

    start_time = time.time()

    # read data
    X_t, y_t = readData('a7a.train', 123)
    X_d, y_d = readData('a7a.dev', 123)
    X_p, y_p = readData('a7a.test', 123)


    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=float,  help='Value of C, if you want to see movement of accuracy over iterations on single C value')
    parser.add_argument('-cm', '--min_c', type=float, help='Minimum value of C, default = 1.0')
    parser.add_argument('-cx', '--max_c', type=float, help='Maximum value of C, default = 100.0')
    parser.add_argument('-s', '--step_length', type=float, help='Step length of C, default = 1.0')
    parser.add_argument('-p', '--processes', type=int,  help='Number of processes, default = 1')
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations, default = 100')
    args = parser.parse_args()


    if args.iterations:
        n_iteration = int(args.iterations)
    else:
        n_iteration = 100




    # test accuracy move on single c value
    if args.c:

        c = float(args.c)
        acc_arr_single_c = []

        # train
        w, b = train(X_t, y_t, c)

        # predict
        print('Test accuracy\t\t%f' % predict(w, b, X_p, y_p))


        end_time = time.time()
        print('%f seconds' % (end_time - start_time))

        plt.plot(acc_arr_single_c)
        plt.savefig('svm_acc')
        plt.show()


    # test accuracy vs. c values
    else:

        if args.min_c:
            min_c = float(args.min_c)
        else:
            min_c = 1.0

        if args.max_c:
            max_c = float(args.max_c)
        else:
            max_c = 100.0

        if args.processes:
            n_process = int(args.processes)
        else:
            n_process = 1

        if args.step_length:
            step_len = float(args.step_length)
        else:
            step_len = 1.0


        n_c = int((max_c - min_c) / step_len) + 1
        done = Value('i', 0)
        c_arr = Array('d', n_c)
        for i in range(n_c):
            c_arr[i] = min_c + step_len * i
        acc_arr = Array('d', n_c)



        # launch processe
        p_arr = []
        for pid in range(n_process):
            p = Process(target=testC, args=(pid, c_arr, acc_arr, X_t, y_t, X_d, y_d, done))
            p_arr.append(p)
            p.start()

        # join processe
        for p in p_arr:
            p.join()



        # get best/worst C and accuracy
        best_c = c_arr[0]
        worst_c = c_arr[0]
        max_acc = acc_arr[0]
        min_acc = acc_arr[0]
        for i in range(n_c):
            if acc_arr[i] > max_acc:
                max_acc = acc_arr[i]
                best_c = c_arr[i]
            if acc_arr[i] < min_acc:
                min_acc = acc_arr[i]
                worst_c = c_arr[i]


        # predict test data
        w, b = train(X_t, y_t, best_c)
        acc = predict(w, b, X_p, y_p)

        print('\nBest C value\t\t%.3f\nMax dev accuracy\t%f\nWorst C value\t\t%.3f\nMin dev accuracy\t%f\nTest accuracy\t\t%f' % (best_c, max_acc, worst_c, min_acc, acc))


        end_time = time.time()
        print('%f seconds' % (end_time - start_time))

        # plot
        plt.plot(c_arr, acc_arr)
        plt.show()
        plt.savefig('c-acc')







