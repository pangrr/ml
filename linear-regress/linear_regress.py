from __future__ import print_function
import threading
import numpy as np 
from numpy import linalg
import sys


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




def train(X, y, lamda):
	return np.dot(linalg.inv(np.dot(X.T, X) + np.identity(X.shape[1], float) * lamda), np.dot(X.T, y))
        




def predict(w, X, y):
	y_p = np.dot(X, w)
	return accur(y, y_p)




    
def accur(y_o, y_p):
	a = 0.0
	for i in range(len(y_o)):
		if y_o[i] * y_p[i] > 0:
			a += 1	
	return a / len(y_o)


        


class testLamdaThread(threading.Thread):
	def __init__(self, lamdas, X_t, y_t, X_d, y_d):
		threading.Thread.__init__(self)
		self.lamdas = lamdas
		self.X_t = X_t
		self.y_t = y_t
		self.X_d = X_d
		self.y_d = y_d
	
	def run(self):
		global MAX_ACC
		global BEST_LAMDA
		global DONE
		global TOTAL
		global LOCK
	
		for lamda in self.lamdas:
			acc = predict(train(self.X_t, self.y_t, lamda), self.X_d, self.y_d)
			LOCK.acquire()
			if(acc > MAX_ACC):
				MAX_ACC = acc
				BEST_LAMDA = lamda
			DONE += 1
			print('testing lamdas %d/%d ...' % (DONE, TOTAL), end='\r')
			LOCK.release()





	
# input arguments
if len(sys.argv) != 5:
	print('arguments error: there must be 4 arguments:\nmin_lamda max_lamda number_of_lamda number_of_threads')
	sys.exit(1)
try:
	min_lamda = float(sys.argv[1])
	max_lamda = float(sys.argv[2])
except ValueError:
	print('arguments error: lamda value should be float')
	sys.exit(1)
try:
	n_lamdas = int(sys.argv[3])
except ValueError:
	print('arguments error: number of lamdas should be integer')
	sys.exit(1)
try:
	n_threads = int(sys.argv[4])
except ValueError:
	print('arguments error: number of threads should be integer')
	sys.exit(1)


# arguemnts check
if max_lamda < min_lamda:
	print('arguments error: max lamda should be no less than min lamda')
	sys.exit(1)
if min_lamda <= 0.0:
	print('arguments error: lamda should be positive value')
	sys.exit(1)
if n_lamdas < 1:
	print('arguments error: number of lamdas should be at least 1')
	sys.exit(1)
if n_threads < 1:
	print('arguments error: number of threads should be at least 1')
	sys.exit(1)
if n_threads > 100:
	print('warning: large number of threads may cause runtime error')




# read data from files
X_t, y_t = readData('/u/cs446/data/adult/a7a.train', 123)
X_d, y_d = readData('/u/cs446/data/adult/a7a.dev', 123)
X_p, y_p = readData('/u/cs446/data/adult/a7a.test', 123)



# initialize global variables
TOTAL = n_lamdas
DONE = 0
BEST_LAMDA = 0.0
MAX_ACC = 0.0
LOCK = threading.Lock()



# special case
if n_lamdas - 2 < 0:
	t = testLamdaThread([min_lamda], X_t, y_t, X_d, y_d)
	t.start()
	t.join()
	print('\n\n%.3f ... best lamda\n%.4f ... best accuracy in dev data prediction\n%.4f ... accuracy in test data prediction' % (BEST_LAMDA, MAX_ACC, predict(train(X_t, y_t, BEST_LAMDA), X_p, y_p)))
	sys.exit(0)




# general cases
step_len = (max_lamda - min_lamda) / float(n_lamdas - 1)	
lamdas_per_thread = n_lamdas / n_threads
	



# partition lamdas
lamdas_arr = []
for i in range(n_threads):
	lamdas = []
	if i < n_threads - 1:
		for j in range(i* lamdas_per_thread, (i + 1) * lamdas_per_thread):
			lamdas.append(min_lamda + (step_len * j))
	else: # last thread
		for j in range(i * lamdas_per_thread, n_lamdas):
			lamdas.append(min_lamda + (step_len * j))
	lamdas_arr.append(lamdas)

# launch threads
threads = []
for lamdas in lamdas_arr:
	t = testLamdaThread(lamdas, X_t, y_t, X_d, y_d)
	threads.append(t)
	t.start()

# join threads
for t in threads:
	t.join()

print('\n\n%.3f ... best lamda\n%.4f ... best accuracy in dev data prediction\n%.4f ... accuracy in test data prediction' % (BEST_LAMDA, MAX_ACC, predict(train(X_t, y_t, BEST_LAMDA), X_p, y_p)))
