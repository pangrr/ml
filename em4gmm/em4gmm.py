# em4gmm.py
# Implement Expectation Maximization Algorithm for Gaussian Mixture Models.

from __future__ import print_function
import numpy as np
from numpy import linalg
import math
import random
import matplotlib
matplotlib.use('Agg') # Comment this line to plot in OS with GUI
import matplotlib.pyplot as plt








class Sample:

    # Read sample data from the given file.
    def __init__ (self, file):
        content = open (file, 'r').readlines ()
        self.nData = len (content)  # Number of dimensions of the sample data.
        self.nDim = len (content[0].split ())   # Number of sample data points.


        self.data = np.zeros ([self.nData, self.nDim])    # Sample data as a 2-D array. Each row represents a data point. Each column represents a dimension.

        r = 0
        for line in open (file, 'r').readlines ():
            dataPoint = line.split ()
            for c in range (self.nDim):
                self.data[r][c] = float (dataPoint[c])
            r += 1











class Gauss:


    def __init__ (self, nDim, mean, cov, mix):
        self.nDim = nDim    # Number of dimensions of this Gaussian distribution.
        self.mean = mean    # Mean array.
        self.cov = cov  # Covariance matrix.
        self.mix = mix  # Mixing coefficient.
        self.newCov = cov    # New covariances computed at M step.
        self.newMean = mean  # New mean computed at M step.
        self.newMix = mix    # New mixing coefficients computed at M step.
        self.pop = 0   # Number of sample data assigned to this Gaussian distribution. Computed in terms of probability.



    # Compute the probability of given data in this Gaussian distribution.
    def prob (self, data):
        A = 1.0 / math.sqrt((2*math.pi)**self.nDim * linalg.det(self.cov))
        B = -1.0/2.0 * np.dot(np.dot((data-self.mean).T, linalg.inv(self.cov)), data-self.mean)
        return A * math.exp (B)



    # Update the parameters.
    def updatePara (self):
        self.mean = self.newMean
        self.cov = self.newCov
        self.mix = self.newMix









class GMM:


    # Initialize the GMM by generating its components.
    def __init__ (self, trainData, devData, nComp, maxIt, covType):
        # Type of covariance matrix.
        # Possible values: "tied" "separate".
        self.covType = covType
        # Maximum number of iterations for EM training.
        self.maxIt = maxIt
        # Training data for EM training.
        self.trainData = trainData
        # Developing data.
        self.devData = devData
        # Number of dimensions.
        self.nDim = trainData.nDim
        # Number of components.
        self.nComp = nComp
        # A matrix of responsibilities computed and updated at every E step
        self.resp = np.zeros([trainData.nData, nComp])
        # A list of Gaussian Distributions.# A list of log likelihood growing during EM training.
        self.comp = []
        # Boolean value whether the log likelihood converges before reaching the maximum number of iterations for EM training
        self.converge = False
        # A list of log likelihood on training data computed for each iteration of EM training.
        self.trainLike = []
        # A list of log likelihood on developing data computed for each iteration of EM training.
        self.devLike = []

        self.initPara ()



    # Initialize parameters for each component.
    # Select nComp random data points as the initial mean for each component..
    # Select the covariance of the whole training data as the covariance for each component.
    def initPara (self):
        randData = self.randData ()
        initCov = self.initCov ()
        initMix = 1.0 / self.nComp

        for k in range (self.nComp):
            gauss = Gauss (self.nDim, randData[k], initCov, initMix)
            self.comp.append (gauss)






    ######## EM training function #########
    def train (self):
        for i in range (self.maxIt):
            print ("Training...%d/%d" % (i+1, self.maxIt), end="\r")

            self.Estep ()
            self.Mstep ()
            self.like ()
#            if self.isConverge (10, 1):
#                self.converge = True
#                break
            self.updatePara ()
        print ()






    ######## EM step functions #######

    # Update parameters for all components.
    def updatePara (self):
        for gauss in self.comp:
            gauss.updatePara ()


    # Compute all responsibilities. i.e. E step.
    def Estep (self):
        for n in range (self.trainData.nData):
            dataPoint = self.trainData.data[n]

            for k in range (self.nComp):
                self.resp[n][k] = self.response (dataPoint, k)






    # Compute and save new parameter values for all components given responsibilities. i.e. M step.
    def Mstep (self):
        for k in range (self.nComp):
            self.pop (k)
            self.newMean (k)
            if self.covType == "tied":
            # Tied covariance matrix as the covariance matrix of the whole training data remains unchanged.
                a = 1   # Nothing useful.
            elif self.covType == "separate":
                self.newCov (k)
            else:
                print ("Error: " + self.covType + " is not a supported covariance matrix type.")
                sys.exit (0)
            self.newMix (k)





    # Compute the log likelihood on training data and developing data based on the new parameters.
    # Add the new log likelihood to the lists.
    def like (self):
        # Training data.
        like = 0.0
        for dataPoint in self.trainData.data:
            log = 0.0
            for gauss in self.comp:
                log += gauss.mix * gauss.prob (dataPoint)

            like += math.log (log)
        self.trainLike.append (like)

        # Developing data.
        like = 0.0
        for dataPoint in self.devData.data:
            log = 0.0
            for gauss in self.comp:
                log += gauss.mix * gauss.prob (dataPoint)

            like += math.log (log)
        self.devLike.append (like)






    # Test log likelihood convergence by computing the variance of the last given number of log likelihood values.
    # Converge if the variance is less than the given threshold.
    def isConverge (self, n, e):
        l = len (self.trainLike)
        if l < n or np.var (self.trainLike[l-n: l+1]) > e:
            return False
        else:
            return True










    ######### Substep functions #########

    # Compute and return a responsibility given a data and component. Used in E step.
    def response (self, dataPoint, k):
        deno = 0.0

        for j in range (self.nComp):
            jgauss = self.comp[j]
            deno += jgauss.mix * jgauss.prob (dataPoint)

        kgauss = self.comp[k]

        return kgauss.mix * kgauss.prob (dataPoint) / deno





    # Compute and update the population of the given component.
    # Used in M step.
    def pop (self, k):
        gauss = self.comp[k]
        gauss.pop = 0
        for n in range (self.trainData.nData):
            gauss.pop += self.resp[n][k]


    # Compute and save the new means for the given component.
    # Used in M step.
    def newMean (self, k):
        gauss = self.comp[k]
        gauss.newMean = np.zeros (self.nDim)
        for n in range (self.trainData.nData):
            gauss.newMean += self.resp[n][k] * self.trainData.data[n]
        gauss.newMean /= gauss.pop


    # Compute and save the new covariances for the given component.
    # Used in M step.
    def newCov (self, k):
        gauss = self.comp[k]
        gauss.newCov = np.zeros ([self.nDim, self.nDim])
        for n in range (self.trainData.nData):
            dataPoint = self.trainData.data[n]
            gauss.newCov += self.resp[n][k] * np.outer((dataPoint-gauss.newMean), (dataPoint-gauss.newMean))
        gauss.newCov /= gauss.pop




    # Compute and save the new mixing coefficient for the given component.
    # Used in M step.
    def newMix (self, k):
        gauss = self.comp[k]
        gauss.newMix = gauss.pop / self.trainData.nData










    # Select and return nComp random data points from the training data.
    def randData (self):
        select = []
        for i in range (self.nComp):
            select.append (self.trainData.data[random.randint (0, self.trainData.nData-1)])
        return select


    # Compute and return the covariance of the whole training data.
    def initCov (self):
        return np.cov (self.trainData.data.T)






# Plot the log likelihood of train data and dev data over iterations for GMMs of different number of components for both tied and separate convariance matrices.

def test (trainData, devData, minNComp, maxNComp, maxIt):
    nComps = list (range(minNComp, maxNComp+1))

    for nComp in range (minNComp, maxNComp+1):
        print ("\nTesting...%d/%d" % (nComp-minNComp+1, maxNComp-minNComp+1))
        gmmTie = GMM (trainData, devData, nComp, maxIt, "tied")
        gmmTie.train ()

        gmmSep = GMM (trainData, devData, nComp, maxIt, "separate")
        gmmSep.train ()

        plt.plot (gmmTie.trainLike, label="tied train")
        plt.plot (gmmSep.trainLike, label="separate train")
        plt.legend (loc="best")
        plt.savefig (str(nComp)+"_train")
        plt.clf ()


        plt.plot (gmmTie.devLike, label="tied dev")
        plt.plot (gmmSep.devLike, label="seperate dev")
        plt.legend (loc="best")
        plt.savefig (str(nComp)+"_dev")
        plt.clf ()




if __name__ == "__main__":
    print ("Reading data...", end="")
    trainData = Sample ("train.dat")
    devData = Sample ("dev.dat")
    print ("Done.")
    test (trainData, devData, 1, 5, 20)








