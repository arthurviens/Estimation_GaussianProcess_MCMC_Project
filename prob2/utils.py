import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats 


def kov(lbda, x, xprime):
    return (1+((x-xprime)/lbda)+((x-xprime)*(x-xprime)/(3*lbda*lbda))) \
        * np.exp(-(x-xprime)/lbda)


def sigmaKov(X, lbda, func):
    sh = X.shape[0]
    sig = np.zeros((sh, sh))
    for i in range(sh):
        valxi = X[i]
        for j in range(sh):
            sig[i, j] = func(lbda, valxi, X[j])
    assert sig.shape == (sh, sh), f"Covariance matrix has a wrong shape {sig.shape} instead of {(sh, sh)}" 
    return sig


def simulGP_gauss(X, N, lbda):
    sigma = sigmaKov(X, lbda, kov)
    L = np.linalg.cholesky(sigma)
    Lt = L.T
    fig, ax = plt.subplots()
    plt.xlabel("X")
    plt.ylabel("Valeur de la variable aléatoire")
    plt.title(r"Réalisations de GP pour $\lambda$ = " + str(lbda))
    for i in range(5):
        g = np.random.normal(0, 1, size=N)
        z = Lt @ g 
        #display(z.round(2))
        ax.plot(X, z, label = f"Sim {i}")
    plt.legend()
    plt.grid()
    plt.show()


def simulGP_sin(X, N, lbda):
    sigma = sigmaKov(X, lbda, kov)
    L = np.linalg.cholesky(sigma)
    Lt = L.T
    fig, ax = plt.subplots()
    plt.xlabel("X")
    plt.ylabel("Valeur de la variable aléatoire")
    plt.title(r"Réalisations de GP pour $\lambda$ = " + str(lbda))
    for i in range(5):
        g = np.random.normal(0, 1, size=N)
        z = Lt @ g 
        #display(z.round(2))
        ax.plot(X, z, label = f"Sim {i}")
    plt.legend()
    plt.grid()
    plt.show()