import numpy as np
import math
import matplotlib.pyplot as plt

def readData(file, n_line, n_feats):
    X = np.zeros([n_line, n_feats+1])
    Y = np.zeros(n_line)
    f = open(file, 'r')
    line = f.readline()
    line_no = 0
    while line:
        line = line.strip()
        fields = line.split()
        Y[line_no] = int(fields[0])
        X[line_no][0] = 1
        for feat in fields[1:]:
            feat_ident = feat.split(':')
            X[line_no][int(feat_ident[0])] = 1
        line_no += 1
        line = f.readline()
    return (X, Y)


def train(X, Y, n_feat, alpha):
    trainsample = len(X)
    p0num = 0
    p1num = 0
    probC1 = 0
    probC0 = 0
    probX1 = np.array(n_feat+1)
    probX0 = np.array(n_feat+1)
    probX0num = np.zeros(len(X[1]))
    probX1num = np.zeros(len(X[1]))

    for i in range(trainsample):
        if Y[i] == 1:
            p1num += 1
            probX1num += X[i]
        else:
            p0num += 1
            probX0num += X[i]

    probC1 = float(p1num) / float(trainsample)
    probC0 = float(p0num) / float(trainsample)
    probX1 = (probX1num + alpha) / (p1num + 2 * alpha)
    probX0 = (probX0num + alpha) / (p0num + 2 * alpha)

    return (probC1, probC0, probX1, probX0)


def classify(probC1, probC0, probX1, probX0, X_new, Y_new):
    numsamples = len(X_new)
    y_hat = np.zeros(len(Y_new))

    for i in range(numsamples):
        probX1i= 1
	probX0i= 1
        xx = X_new[i]

        for k in range(len(X_new[1])):
            if xx[k] ==1:
                probX1i = probX1[k] * probX1i
                probX0i = probX0[k] * probX0i
            else:
                probX1i = (1 - probX1[k]) * probX1i
                probX0i = (1 - probX0[k]) * probX0i

        probC1i = (float(probX1i) * float(probC1))
        probC0i = (float(probX0i) * float(probC0))

        if probC1i > probC0i:
            y_hat[i] = 1
        else:
            y_hat[i] = -1

    return accuracy(y_hat,Y_new)


def accuracy(y_hat,Y_new):
    i = 0
    rr = 0
    while i < len(Y_new):
       if (Y_new[i] * y_hat[i])>0:
           rr += 1
       i += 1
    acc = float(rr)/float(len(Y_new))
    return acc


if __name__ == '__main__':
    X_dev, Y_dev = readData('./a7a.dev', 8000, 123)
    X_train, Y_train = readData('./a7a.train', 16100, 123)
    X_test ,Y_test= readData('./a7a.test', 8462, 123)

    #set test alpha from 1 to n
    n = 5
    acc  = np.zeros(n)

    for a in range(1, n+1):
        probC1,probC0,probX1,probX0 = train(X_train, Y_train, 123, a)
        acc[a-1] = classify(probC1, probC0, probX1, probX0, X_dev, Y_dev)

    #find best alpha and max accuracy
    best_alpha = 0.0
    max_acc = 0.0
    for a in range(1, n+1):
        if acc[a-1] > max_acc:
            max_acc = acc[a-1]
            best_alpha = a

    #predict test data
    pre_acc = classify(probC1, probC0, probX1, probX0, X_test, Y_test)

    #plot
    plt.plot(range(1, n+1), acc)
    plt.show()

    print 'best alpha: %.3f\nmax accuracy: %.3f\ntest accuracy: %.3f' % (best_alpha, max_acc, pre_acc)





