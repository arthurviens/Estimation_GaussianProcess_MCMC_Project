import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats 

def kernel_ISE(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def kernel(X1, X2, lbda):
    """
    Project subject kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    a = np.abs(X1 - X2.reshape(-1, 1))
    aa = np.square(a)
    return (1 + (a / lbda) + (aa / (3*lbda*lbda))) * np.exp(-a/lbda)



def sigmaKov(X1, X2, lbda, func=kernel, add_noise=True):
    sh = X1.shape[0]
    assert X1.shape[0] == X2.shape[0], "X1 and X2 are not of same size"
    sig = func(X1, X2, lbda)
    
    if add_noise:
        sig = sig + np.diag(np.random.uniform(low = 0., high = 0.000001, size=sh))
    assert sig.shape == (sh, sh), f"Covariance matrix has a wrong shape {sig.shape} instead of {(sh, sh)}" 
    return sig



def simulGP_gauss(X, N, lbda):
    sigma = sigmaKov(X, X, lbda)
    L = np.linalg.cholesky(sigma)
    #Lt = L.T
    fig, ax = plt.subplots()
    plt.xlabel("X")
    plt.ylabel("Valeur de la variable aléatoire")
    plt.title(r"Réalisations de GP pour $\lambda$ = " + str(lbda))
    for i in range(5):
        g = np.random.normal(0, 1, size=N)
        z = L @ g 
        #display(z.round(2))
        ax.plot(X, z, label = f"Sim {i}")
    plt.legend()
    plt.grid()
    plt.show()


def simulGP_sin(X, N, lbda):
    X_random = draw_uniform_in_intervals(N)
    Xsin = np.sin(4*np.pi*X_random)
    sigma = sigmaKov(X, X, lbda)
    L = np.linalg.cholesky(sigma)
    #Lt = L.T
    fig, ax = plt.subplots()
    plt.xlabel("X")
    plt.ylabel("Valeur de la variable aléatoire")
    plt.title(r"Réalisations de GP pour $\lambda$ = " + str(lbda))
    for i in range(2):
        g = np.random.normal(size=N)
        
        z = L @ g 
        #display(z.round(2))
        ax.plot(X_random, z, label = f"Sim {i}")
    ax.plot(X_random, Xsin, label = "Sinus")
    plt.legend()
    plt.grid()
    plt.show()


def draw_uniform_in_intervals(N):
    X = np.linspace(0, 1, N+1)
    numbers = np.zeros(N)
    for i in range(N):
        tup = X[i:i+2]
        numbers[i] = np.random.uniform(low = tup[0], high = tup[1])
    return numbers


def neglikelihood(lbda, X, Z):
    if lbda <= 0:
        lbda = 1e-7
    sigma = sigmaKov(X, X, lbda)
    logdet = np.linalg.slogdet(sigma)[1]
    invsig = np.linalg.inv(sigma)
    
    d = len(X)
    Z = Z.reshape(d, 1)
    #print(f"Z : {Z}")
    Zt = Z.T
    #print(f"Shapes  zt {Zt.shape}, invsig {invsig.shape}, z {Z.shape},")
    seghalf = Zt @ invsig @ Z
    return 0.5 * (d * np.log(2 * np.pi) + logdet + seghalf)