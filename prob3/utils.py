import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats 
from scipy.special import factorial
#import math



class ParametrizedMCMC:
    """
    class to define gamma(x) and posterior(x) functions of which the 
    gamma law is parametrized accordingly
    """
    def __init__(self, x, alpha, beta):
        assert isinstance(x, int), "x is not an integer"
        assert alpha > 0, "Alpha parameter must be > 0 "
        assert beta > 0, "Beta parameter must be > 0 "
        self.x = x
        self.alpha = alpha
        self.beta = beta

    def getGamma(self):
        return scipy.stats.gamma(self.alpha, loc=0, scale = 1 / self.beta).pdf

    def getPoissonLikelihood(self):
        return lambda theta: np.power(theta, self.x) * np.exp(-theta) / factorial(self.x, exact=True)

    def getPosterior(self):
        likelihood = self.getPoissonLikelihood()
        prior = self.getGamma()
        return lambda theta: likelihood(theta) * prior(theta)

    def getRealPosterior(self):
        return scipy.stats.gamma(self.alpha + self.x, loc=0, scale = 1 / (self.beta + 1)).pdf


def MCMC_sampling(distribution_parameters, N, s):
    #----------------------------------------------------------------------------------------#
    # Metropolis Hastings sampling from the posterior distribution

    # Generate the posterior function as product of prior and likelihood
    posterior = distribution_parameters.getPosterior()


    theta = 0
    p = posterior(theta)

    samples = []

    for i in range(N):
        # Proposal
        thetan = theta + np.random.normal()
        # Posterior
        pn = posterior(thetan)
        if pn >= p:
            p = pn
            theta = thetan
        else:
            u = np.random.rand()
            if u < pn/p:
                p = pn
                theta = thetan
        if i % s == 0:
            samples.append(theta)

    samples = np.array(samples[int(len(samples)/2):])
    print(f"Samples shape : {samples.shape}. Generated from {N} steps, \
        taking 1 sample out of {s}, and only the last half of the selected samples") 
    return samples