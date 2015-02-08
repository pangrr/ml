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



    # def __init__(self, X_, Y_, n_feat, alpha_ = 0.0):
        # self.X = X_
        # self.Y = Y_
        # self.alpha = alpha_
        # self.pc1 = 0
        # self.pc0 = 0
        # self.pwc1 = np.array(n_feat+1)
        # self.pwc0 = np.array(n_feat+1)
        # self.pw = np.array(n_feat+1)

def trainNB0(X,Y,n_feat,alpha_=0.0):
    numTrainsamples = len(X)
    p0num = 0 ; p1num = 0;
    pc1 = 0
    pc0 = 0
    pwc1 = np.array(n_feat+1)
    pwc0 = np.array(n_feat+1)
    pw = np.array(n_feat+1)
    alpha = alpha_
    pwc0num = np.zeros(len(X[1]))
    pwc1num = np.zeros(len(X[1]))

    for i in range(numTrainsamples):
        if Y[i] == 1:
            p1num += 1
            pwc1num += X[i]
        else:
            p0num += 1
            pwc0num += X[i]
    pc1 = float(p1num)/float(numTrainsamples)
    pc0 = float(p0num)/float(numTrainsamples)

    pwc1 = (pwc1num+alpha)/(p1num+2*alpha)
    pwc0 = (pwc0num+alpha)/(p0num+2*alpha)
    pw = X.sum(axis= 0)/(numTrainsamples)
    return (pc1,pc0,pwc1,pwc0,pw)


def classifyNB(pc1,pc0,pwc1,pwc0,pw,X_new,Y_new):
    numsamples = len(X_new)
    y_hat = np.zeros(len(Y_new))
    for i in range(numsamples):
        pwc1i= 1;pwc0i= 1;pwi = 1
        xx = X_new[i]

        for k in range(len(X_new[1])):
            if xx[k] ==1:
                pwc1i *= pwc1[k]
                pwc0i *= pwc0[k]
            else:
                pwc1i *= (1-pwc1[k])
                pwc0i *= (1-pwc0[k])


        pc1i = (float(pwc1i) * float(pc1))#/(float(pwi))
        pc0i = (float(pwc0i) * float(pc0))#/(float(pwi))
        if pc1i > pc0i:
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

##        for i in range(len(Y_new)):
##            if (y_hat[i] * Y_new[i]) >0:
##                rr +=1
##        acc = float(rr)/float(len(Y_new))
##        return acc

if __name__ == '__main__':
    X_dev, Y_dev = readData('./a7a.dev', 8000, 123)
    X_train, Y_train = readData('./a7a.train', 16100, 123)
    X_test ,Y_test= readData('./a7a.test', 8462, 123)
    acc  = np.zeros(1000);alpha = np.zeros(1000)
    acc1 = 0
    pc1,pc0,pwc1,pwc0,pw = trainNB0(X_train, Y_train,123,0)
    acc = classifyNB(pc1,pc0,pwc1,pwc0,pw,X_dev,Y_dev)


    # classifi1 = naivebayes(X_train, Y_train,123,303)
    # classifi1.trainNB0(X_train, Y_train)

    # acc1 = classifi1.classifyNB(X_test, Y_test)
    print acc

    # plt.plot(alpha,acc)
    # plt.show()




