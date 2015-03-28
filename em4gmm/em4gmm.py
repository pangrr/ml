# em4gmm.py
# Implement Expectation Maximization Algorithm for Gaussian Mixture Models.

import numpy as np
from numpy import linalg
import math
import random
import matplotlib
#matplotlib.use('Agg') # Comment this line to plot in OS with GUI
import matplotlib.pyplot as plt
import time











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
            for c in range (0, self.nDim):
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
    def __init__ (self, sample, nComp, maxIt):
        self.maxIt = maxIt  # Maximum number of iterations for EM training.
        self.sample = sample    # Sample data for EM training.
        self.nDim = sample.nDim  # Number of dimensions.
        self.nComp = nComp  # Number of components.
        self.resp = np.zeros([sample.nData, nComp])  # A matrix of responsibilities computed and updated at every E step
        self.comp = []  # A list of Gaussian Distributions.# A list of log likelihood growing during EM training.
        self.converge = False   # Boolean value whether the log likelihood converges before reaching the maximum number of iterations for EM training
        self.hisLike = [] # A list of log likelihood growing at each iteration of EM training.

        self.initPara ()



    # Initialize parameters for each component.
    # Select nComp random data points as the initial mean for each component..
    # Select the covariance of the whole sample as the covariance for each component.
    def initPara (self):
        randData = self.randData ()
        sampleCov = self.sampleCov ()
        initMix = 1.0 / self.nComp

        for k in range (0, self.nComp):
            gauss = Gauss (self.nDim, randData[k], sampleCov, initMix)
            self.comp.append (gauss)






    ######## EM training function #########
    def train (self):
        for i in range (0, self.maxIt):
            print ("Training...%d/%d" % (i+1, self.maxIt), end = "\r")

            self.Estep ()
            self.Mstep ()
            self.like ()
            if self.isConverge (10, 1):
                self.converge = True
                break
            self.updatePara ()






    ######## EM step functions #######

    # Update parameters for all components.
    def updatePara (self):
        for gauss in self.comp:
            gauss.updatePara ()


    # Compute all responsibilities. i.e. E step.
    def Estep (self):
        for n in range (0, self.sample.nData):
            dataPoint = self.sample.data[n]

            for k in range (0, self.nComp):
                self.resp[n][k] = self.response (dataPoint, k)






    # Compute and save new parameter values for all components given responsibilities. i.e. M step.
    def Mstep (self):
        for k in range (0, self.nComp):
            self.pop (k)
            self.newMean (k)
            self.newCov (k)
            self.newMix (k)





    # Compute the log likelihood based on the new parameters.
    # Add the new log likelihood to the list.
    def like (self):
        like = 0.0
        for dataPoint in self.sample.data:
            log = 0.0
            for gauss in self.comp:
                log += gauss.mix * gauss.prob (dataPoint)

            like += math.log (log)

        self.hisLike.append (like)






    # Test log likelihood convergence by computing the variance of the last given number of log likelihood values.
    # Converge if the variance is less than the given threshold.
    def isConverge (self, n, e):
        l = len (self.hisLike)
        if l < n or np.var (self.hisLike[l-n: l+1]) > e:
            return False
        else:
            return True










    ######### Substep functions #########

    # Compute and return a responsibility given a data and component. Used in E step.
    def response (self, dataPoint, k):
        deno = 0.0

        for j in range (0, self.nComp):
            jgauss = self.comp[j]
            deno += jgauss.mix * jgauss.prob (dataPoint)

        kgauss = self.comp[k]

        return kgauss.mix * kgauss.prob (dataPoint) / deno





    # Compute and update the population of the given component. Used in M step.
    def pop (self, k):
        gauss = self.comp[k]
        gauss.pop = 0
        for n in range (0, self.sample.nData):
            gauss.pop += self.resp[n][k]


    # Compute and save the new means for the given component. Used in M step.
    def newMean (self, k):
        gauss = self.comp[k]
        gauss.newMean = np.zeros (self.nDim)
        for n in range (0, self.sample.nData):
            gauss.newMean += self.resp[n][k] * self.sample.data[n]
        gauss.newMean /= gauss.pop


    # Compute and save the new covariances for the given component. Used in M step.
    def newCov (self, k):
        gauss = self.comp[k]
        gauss.newCov = np.zeros ([self.nDim, self.nDim])
        for n in range (0, self.sample.nData):
            dataPoint = self.sample.data[n]
            gauss.newCov += self.resp[n][k] * np.outer((dataPoint-gauss.newMean), (dataPoint-gauss.newMean))
        gauss.newCov /= gauss.pop




    # Compute and save the new mixing coefficient for the given component, Used in M step.
    def newMix (self, k):
        gauss = self.comp[k]
        gauss.newMix = gauss.pop / self.sample.nData










    # Select and return nComp random data points from the sample.
    def randData (self):
        select = []
        for i in range (0, self.nComp):
            select.append (self.sample.data[random.randint (0, self.sample.nData)])
        return select


    # Compute and return the covariance of the whole sample.
    def sampleCov (self):
        return np.cov (self.sample.data.T)








# Plot the log likelihood of the trained GMMs of different number of components.
def testNComp (sample, minNComp, maxNComp, maxIt):
    nComps = list (range(minNComp, maxNComp+1))
    likes = []  # Store the log likelihood of the trained GMMs of different number of components.

    for nComp in range (minNComp, maxNComp+1):
        print ("\nTesting...%d/%d" % (nComp, maxNComp-minNComp+1))
        gmm = GMM (sample, nComp, maxIt)
        gmm.train ()
        likes.append (gmm.hisLike[-1])

    # Plot.
    plt.plot (nComps, likes)
    plt.show ()



# Plot the path of log likelihood during the training of a GMM
def trainGMM (sample, nComp, maxIt):
    print ("Initializing model...", end="")
    gmm = GMM (sample, 5, 100)
    print ("Done.")
    gmm.train ()


    if gmm.converge:
        print ("\nConverge.")
    else:
        print ("\nNot converge.")

    # Plot
    plt.plot (gmm.hisLike)
    plt.show ()





if __name__ == '__main__':
    print ("Reading sample...", end="")
    sample = Sample ('points.dat')
    print ("Done.")
    testNComp (sample, 1, 10, 100)
