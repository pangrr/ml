# em4hmm.py
# Implement Expectation Maximization Algorithm for Hidden Markov Models.

from __future__ import print_function
import numpy as np
from numpy import linalg
import math
import random
import matplotlib
matplotlib.use('Agg') # Comment this line to plot in OS with GUI
import matplotlib.pyplot as plt








class DataSet:

    # Read data set from the given file.
    def __init__ (self, file):
        content = open (file, 'r').readlines ()
        self.nData = len (content)  # Number of dimensions of the data set.
        self.dim = len (content[0].split ())   # Number of data points.


        self.data = np.zeros ([self.nData, self.dim])    # Data set as a 2-D array. Each row represents a data point. Each column represents a dimension.

        r = 0
        for line in open (file, 'r').readlines ():
            dataPoint = line.split ()
            for c in range (self.dim):
                self.data[r][c] = float (dataPoint[c])
            r += 1




#############################################################################






class Gauss:

    def __init__ (self, dim, mean, cov):
        self.dim = dim    # Number of dimensions of this Gaussian distribution.
        self.mean = mean    # Mean array.
        self.cov = cov  # Covariance matrix.
        self.freq = 0.0 # Number of data points in this gauss.


    # Compute the probability of given data in this Gaussian distribution.
    def prob (self, dataPoint):
        A = math.sqrt ((2*math.pi)**self.dim * linalg.det(self.cov))
        sub = dataPoint - self.mean
        B = - np.dot (np.dot(sub.T, linalg.inv(self.cov)), sub) / 2.0
        return math.exp (B) / A





#############################################################################




class HMM:

    # Initialize the HMM by generating its components.
    def __init__ (self, trainData, devData, nComp, it):
        # Constant fields. #

        # Maximum number of iterations for EM training.
        self.it = it
        # Training data for EM training.
        self.trainData = trainData
        # Developing data.
        self.devData = devData
        # Number of dimensions.
        self.dim = trainData.dim
        # Number of components.
        self.nComp = nComp


        # Parameters of HMM. #

        # Transition matrix.
        self.tran = np.zeros ([nComp, nComp])
        # Initial state distribution.
        self.init = np.zeros (nComp)
        # A list of Gaussian Distributions.
        self.comp = []

        self.initPara ()



        # Intermediate fields. #

        # A matrix of marginal posterior distribution of latent variables computed and updated at every E step.
        self.margin = np.zeros ([nComp, self.trainData.nData])
        # A matrix of joint posterior distribution of two successive latent variables, computed and updated at each E step.
        self.joint = np.zeros ([nComp, nComp, self.trainData.nData-1])
        # A matrix computed in the forward procedure for train data.
        self.fwd = np.zeros ([nComp, self.trainData.nData])
        # A matrix computed in the forward procedure for the dev data.
        self.devfwd = np.zeros([nComp, self.devData.nData])
        # A matrix computed in the backward procedure for training data.
        self.bwd = np.zeros ([nComp, self.trainData.nData])
        # A list representing the adjustment coefficient for fwd and bwd.
        self.adj = np.zeros (self.trainData.nData)
        # A list representing the adjustment coefficient for fwd and bwd on dev data.
        self.devadj = np.zeros(self.devData.nData)


        # A list of log likelihood on training data computed for each iteration during the training.
        self.trainLike = []
        # A list of log likelihood on developing data computed for each iteration during the training.
        self.devLike = []





    # Initialzie parameters of HMM.
    def initPara (self):
        nComp = self.nComp

        for i in range (nComp):
            for j in range (nComp):
                self.tran[i][j] = 1.0 / nComp

        for i in range (nComp):
            self.init[i] = 1.0 / nComp

        self.initComp ()





    # Initialize parameters for each component.
    # Select nComp random data points as the initial mean for each component..
    # Select the covariance of the whole training data as the covariance for each component.
    def initComp (self):
        randData = self.randData ()
        initCov = self.initCov ()

        for i in range (self.nComp):
            gauss = Gauss (self.dim, randData[i], initCov)
            self.comp.append (gauss)


    # Select and return nComp random data points from the training data.
    def randData (self):
        select = []
        for i in range (self.nComp):
            select.append (self.trainData.data[random.randint (0, self.trainData.nData-1)])
            #select.append (self.trainData.data[i])
        return select



    # Compute and return the covariance of the whole training data.
    def initCov (self):
        return np.cov (self.trainData.data.T)


    def train (self):
        for i in range (self.it):
            print ("Training...%d/%d" % (i+1, self.it), end="\r")
            self.EStep ()
            self.MStep ()
        print ()








    # Use forward and backward algorithm to compute marginal distribution and joint distribution.
    def EStep (self):
        self.forward (self.trainData, self.fwd, self.adj)
        self.backward ()
        self.like(self.adj, self.trainLike)
        self.forward(self.devData, self.devfwd, self.devadj)
        self.like(self.devadj, self.devLike)
        self.updateMargin ()
        self.updateJoint ()



    # Forward procedure.
    def forward (self, dataSet, fwd, adj):
        init = self.init
        comp = self.comp
        nComp = self.nComp
        tran = self.tran
        nData = dataSet.nData

        adj[0]= 0.0
        for i in range (nComp):
            fwd[i][0] = init[i] * comp[i].prob(dataSet.data[0])
            adj[0] += fwd[i][0]    # Compute adjustment for fwd.
        for i in range (nComp):
            fwd[i][0] /= adj[0]    # Adjust fwd to prevent it becoming too small.

        for t in range  (1, nData):
            adj[t] = 0.0

            for j in range (nComp):
                fwd[j][t] = 0.0
                for i in range (nComp):
                    fwd[j][t] += fwd[i][t-1] * tran[i][j]
                fwd[j][t] *= comp[j].prob (dataSet.data[t])
                adj[t] += fwd[j][t]

            for j in range (nComp):
                fwd[j][t] /= adj[t]







    # Backward procedure.
    def backward (self):
        dataSet = self.trainData
        nData = dataSet.nData
        bwd = self.bwd
        nComp = self.nComp
        comp = self.comp
        tran = self.tran
        adj = self.adj

        for i in range (nComp):
            bwd[i][nData-1] = 1.0 / adj[nData-1]

        for t in range (nData-2, -1, -1):
            for i in range (nComp):
                bwd[i][t] = 0.0
                for j in range (nComp):
                    bwd[i][t] += bwd[j][t+1] * tran[i][j] * comp[j].prob(dataSet.data[t+1])
                bwd[i][t] /= adj[t+1]










    # Compute marginal distribution.
    def updateMargin (self):
        margin = self.margin
        fwd = self.fwd
        bwd = self.bwd

        for t in range (self.trainData.nData):
            for i in range (self.nComp):
                margin[i][t] = fwd[i][t] * bwd[i][t]




    # Compute joint distribution.
    def updateJoint (self):
        joint = self.joint
        fwd = self.fwd
        adj = self.adj
        bwd = self.bwd
        tran = self.tran
        nComp = self.nComp
        dataSet = self.trainData
        comp = self.comp

        for t in range (dataSet.nData-1):
            dataPoint = dataSet.data[t+1]
            for i in range (nComp):
                for j in range(nComp):
                    joint[i][j][t] = fwd[i][t] * comp[i].prob(dataPoint) * tran[i][j] * bwd[i][t+1] / adj[t+1]




    # Compute and save new parameter values for all components given responsibilities.
    def MStep (self):
        self.updateInit ()
        self.updateTran ()
        self.updateComp ()

        #for gauss in self.comp:
            #print (gauss.cov)





    # Compute and update initial state distribution.
    def updateInit (self):
        for i in range (self.nComp):
            self.init[i] = 0.0
            for j in range (self.nComp):
                self.init[i] += self.margin[j][0]
            self.init[i] = self.margin[i][0] / self.init[i]




    # Compute and update transition matrix.
    def updateTran (self):
        for i in range (self.nComp):
            for j in range (self.nComp):
                sum1 = 0.0
                for t in range (self.trainData.nData-1):
                    sum1 += self.joint[i][j][t]
                sum2 = 0.0
                for k in range (self.nComp):
                    for t in range (self.trainData.nData-1):
                        sum2 += self.joint[i][k][t]
                self.tran[i][j] = sum1 / sum2







    # Compute and update parameters of each component gaussian distribution.
    def updateComp (self):
        for i in range (self.nComp):
            self.freq ()
            self.updateGaussMean (i)
            self.updateGaussCov (i)





    # Compute frequency for each component.
    def freq (self):
        for i in range (self.nComp):
            gauss = self.comp[i]
            gauss.freq = 0.0
            for t in range (self.trainData.nData):
                gauss.freq += self.margin[i][t]






    # Compute and update mean of given gaussian distribution.
    def updateGaussMean (self, i):
        dataSet = self.trainData
        margin = self.margin
        gauss = self.comp[i]

        sum = np.zeros (self.dim)
        for t in range (dataSet.nData):
            sum += margin[i][t] * dataSet.data[t]
        gauss.mean = sum / gauss.freq


    # Compute and update cov of given gaussian distribution.
    def updateGaussCov (self, i):
        dim = self.dim
        dataSet = self.trainData
        margin = self.margin
        gauss = self.comp[i]
        sum = np.zeros ([self.dim, self.dim])

        for t in range (dataSet.nData):
            sub = dataSet.data[t] - gauss.mean
            sum += margin[i][t] * np.outer(sub, sub)
        gauss.cov = sum / gauss.freq






    # Compute and add to record the log likelihood on given data set.
    def like (self, adj, likeList):
        logLike = 0.0
        for i in adj:
            logLike += math.log(i)

        likeList.append(logLike)










###############################################################################







# Plot the log likelihood of train data and dev data over iterations for HMMs of different number of components (latent states).

def test (trainData, devData, minNComp, maxNComp, it):
    nComps = list (range(minNComp, maxNComp+1))

    for nComp in range (minNComp, maxNComp+1):
        print ("\nTesting...%d/%d" % (nComp-minNComp+1, maxNComp-minNComp+1))
        hmm = HMM (trainData, devData, nComp, it)
        hmm.train ()

        fig = plt.figure()
        plt.title("Log Likelihood for " + str(nComp) + " Hidden States")
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Train Data Log Likelihood")
        ax1.plot(hmm.trainLike, label="train", color="red")
        ax1.legend(loc=3)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Dev Data Log Likelihood")
        ax2.plot(hmm.devLike, label="dev")
        ax2.legend(loc=4)
        plt.savefig(str(nComp))
        plt.clf ()





if __name__ == "__main__":
    print ("Reading data...", end="")
    trainData = DataSet("train.dat")
    devData = DataSet("dev.dat")
    print("Done.")
    test(trainData, devData, 1, 5, 10)








