import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats 
from scipy.special import factorial
#import math
from sklearn.neighbors import KernelDensity



class ParametrizedMCMC:
    """
    class to define priors, likelihood posteriors functions when x, alpha
    and beta are fixed
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


    theta = 1
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


def KDE_plot(samples, x_array, y_array):
    density_param = {"density": True}

    # ----------------------------------------------------------------------
    # Plot a 1D density example
    fig, ax = plt.subplots()
    ax.fill(x_array, y_array, fc="black", alpha=0.2, label="real distribution")
    colors = ["navy", "cornflowerblue", "darkorange"]
    kernels = ["gaussian", "tophat", "epanechnikov"]
    lw = 2

    for color, kernel in zip(colors, kernels):
        kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(samples.reshape(-1, 1))
        log_dens = kde.score_samples(x_array.reshape(-1, 1))
        ax.plot(
            x_array,
            np.exp(log_dens),
            color=color,
            lw=lw,
            linestyle="-",
            label="kernel = '{0}'".format(kernel),
        )


    ax.legend(loc="upper right")
    plt.xlabel("Domain of definition")
    plt.ylabel("Density")
    #ax.plot(x_array, -0.005 - 0.01 * np.random.random(x_array.shape[0]), "+k")

    plt.show()