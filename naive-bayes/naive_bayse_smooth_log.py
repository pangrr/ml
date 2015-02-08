from __future__ import print_function
import numpy as np
from numpy import linalg
import sys
import matplotlib
#plot without X window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import collections
from multiprocessing import Process, Array, Value




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
    #C[0][0]=c(y>0), C[1][0]=c(y<0), C[2][0]=c(y>0)+c(y<0)
    #C[0][j+1]=c(xj=1|y>0), C[1][j+1]=c(xj=1|y<0)
    #C[0][end]=c(sum(xj=1)|y>0), C[1][end]=c(sum(xj=1)|y<0)
    #C[2][j+1]=c(xj=1)
    X_n = X.shape[1]
    C = np.zeros([3, X_n+1])
    for i in range(len(y)):
        if y[i] > 0:
                C[0][0] += 1
        else:
                C[1][0] += 1

        for j in range(1, X_n):
            if y[i] > 0:
                C[0][j] += X[i][j]
            else:
                C[1][j] += X[i][j]

    C[2][0] = C[0][0] + C[1][0]
    for i in range(1, X_n):
        C[0][X_n] += C[0][i]
        C[1][X_n] += C[1][i]
        C[2][i] += C[0][i] + C[1][i]
        C[2][X_n] += C[2][i]
    #debug
    #plt.plot(range(90, X_n-1), C[0][90:X_n-1], 'g^', range(90, X_n-1), C[1][90:X_n-1], 'bs')
    #plt.show()
    #print("X_m=%d, X_n=%d, y_n=%d\nc(y>0)=%d, c(y<0)=%d, c(y)=%d\nc(x|y>0)=%d, c(x|y<0)=%d, c(x)=%d" % (X.shape[0], X.shape[1], len(y), C[0][0], C[1][0], C[2][0], C[0][X_n], C[1][X_n], C[2][X_n]))
    return C



def predict(C, alpha, X, y):
    y_p = np.zeros(y.shape[0])
    X_n = X.shape[1]

    for i in range(y.shape[0]):
        pyp = math.log(float(C[0][0]) / float(C[2][0]))
        pyn = math.log(float(C[1][0]) / float(C[2][0]))

        #debug
#        print("pyp %.3f" %(pyp))
#        print("pyn %.3f" %(pyn))


        for j in range(1, X_n):
            if X[i][j] == 0:
                pyp += math.log((float(C[0][0]) - float(C[0][j]) + alpha) / (float(C[0][0]) + alpha * 2))
                pyn += math.log((float(C[1][0]) - float(C[1][j]) + alpha) / (float(C[1][0]) + alpha * 2))
            if X[i][j] == 1:
                pyp += math.log((float(C[0][j]) + alpha) / (float(C[0][0]) + alpha * 2))
                pyn += math.log((float(C[1][j]) + alpha) / (float(C[1][0]) + alpha * 2))
           #debug
#            print("pyp %.3f" %(pyp))
#            print("pyn %.3f" %(pyn))

        if pyp > pyn:
            y_p[i] = 1
        else:
            y_p[i] = -1
    #debug
#    print("alpha=%.3f" % (alpha))
#    print(y_p)
#    print(y)
#    print(y.shape)
#    print(y_p.shape)
    return accur(y, y_p)




def accur(y_o, y_p):
	match = 0.0
	for i in range(y_o.shape[0]):
		if y_o[i] * y_p[i] > 0:
			match += 1
        #debug
#        print("accuracy=%.3f" % (match/len(y_o)))
	return match / len(y_o)




def testAlpha(pi, C, alphaArr, accArr, done, X_d, y_d):
    global total
    global n_processes

    if pi == n_processes - 1:
        for i in range(pi*len(alphaArr)/n_processes, len(alphaArr)):
            acc = predict(C, alphaArr[i], X_d, y_d)
            accArr[i] = acc
            done.value = done.value + 1
            print('testing alphas %d/%d...' % (done.value, total), end='\r')
    else:
        for i in range(pi*len(alphaArr)/n_processes, (pi+1)*len(alphaArr)/n_processes):
            acc = predict(C, alphaArr[i], X_d, y_d)
            accArr[i] = acc
            done.value = done.value + 1
            print('testing alphas %d/%d...' % (done.value, total), end='\r')





# main
start = time.time()


# input arguments
if len(sys.argv) != 5:
	print('arguments error: there must be 4 arguments:\nmin_alpha max_alpha number_of_alpha number_of_threads')
	sys.exit(1)
try:
	min_alpha = float(sys.argv[1])
	max_alpha = float(sys.argv[2])
except ValueError:
	print('arguments error: alpha value should be float')
	sys.exit(1)
try:
	n_alphas = int(sys.argv[3])
except ValueError:
	print('arguments error: number of alphas should be integer')
	sys.exit(1)
try:
	n_processes = int(sys.argv[4])
except ValueError:
	print('arguments error: number of processes should be integer')
	sys.exit(1)


# arguemnts check
if max_alpha < min_alpha:
	print('arguments error: max alpha should be no less than min alpha')
	sys.exit(1)
if min_alpha <= 0.0:
	print('arguments error: alpha should be positive value')
	sys.exit(1)
if n_alphas < 1:
	print('arguments error: number of alphas should be at least 1')
	sys.exit(1)
if n_processes < 1:
	print('arguments error: number of processes should be at least 1')
	sys.exit(1)
if n_processes > 100:
	print('warning: large number of processes may cause runtime error')




# read data from files
X_t, y_t = readData('../adult/a7a.train', 123)
X_d, y_d = readData('../adult/a7a.dev', 123)
X_p, y_p = readData('../adult/a7a.test', 123)


# initialize global variables
total = n_alphas
done = Value('i', 0)
alphaArr = Array('d', n_alphas)
accArr = Array('d', n_alphas)


# train
print("training...")
C = train(X_t, y_t)


# special case
if n_alphas - 2 < 0:
        alphaArr[0] = min_alpha
	p = Process(target=testAlpha, args=(0, C, alphaArr, accArr, done, X_d, y_d))
	p.start()
	p.join()
	print("\n%.3f ... best alpha\n%.4f ... best accuracy in dev data prediction\n%.4f ... accuracy in test data prediction" % (min_alpha, accArr[0], predict(C, min_alpha, X_p, y_p)))
	sys.exit(0)





# general cases
step_len = (max_alpha - min_alpha) / float(n_alphas - 1)
for i in range(n_alphas):
    alphaArr[i] = min_alpha + step_len * i


# launch threads
processes = []
for i in range(n_processes):
	p = Process(target=testAlpha, args=(i, C, alphaArr, accArr, done, X_d, y_d))
	processes.append(p)
	p.start()

# join threads
for p in processes:
	p.join()


# search for best alpha and accuracy
best_alpha = alphaArr[0]
max_acc = accArr[0]
for i in range(len(alphaArr)):
    if accArr[i] > max_acc:
        max_acc = accArr[i]
        best_alpha = alphaArr[i]

end = time.time()

# show results
print("\n%.3f... best alpha\n%.4f... best accuracy in dev data prediction\n%.4f... accuracy in test data prediction" % (best_alpha, max_acc, predict(C, best_alpha, X_p, y_p)))

# plot
plt.plot(alphaArr, accArr)
plt.show()
plt.savefig('result')

print("time elapse: %.3f seconds (%.3f seconds per alpha)" % (end-start, (end-start)/float(n_alphas)))
