import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats 


def kov(lbda, x, xprime):
    return (1+(np.abs(x-xprime)/lbda)+(np.abs(x-xprime)**2 / (3*lbda*lbda))) \
        * np.exp(-np.abs(x-xprime)/lbda)


def sigmaKov(X, lbda, func):
    sh = X.shape[0]
    sig = np.zeros((sh, sh))
    for i in range(sh):
        valxi = X[i]
        for j in range(sh):
            sig[i, j] = func(lbda, valxi, X[j])
    
    sig = sig + np.diag(np.random.uniform(low = 0., high = 0.000001, size=sh))
    assert sig.shape == (sh, sh), f"Covariance matrix has a wrong shape {sig.shape} instead of {(sh, sh)}" 
    return sig


def simulGP_gauss(X, N, lbda):
    sigma = sigmaKov(X, lbda, kov)
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
    X_random = draw_uniform_in_intervals(N+1)
    Xsin = np.sin(4*np.pi*X_random)
    sigma = sigmaKov(Xsin, lbda, kov)
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
    X = np.linspace(0, 1, N)
    numbers = np.zeros(N - 1)
    for i in range(N - 1):
        tup = X[i:i+2]
        numbers[i] = np.random.uniform(low = tup[0], high = tup[1])
    return numbers


def neglikelihood(lbda, Z):
    sigma = sigmaKov(Z, lbda, kov)
    L = np.linalg.cholesky(sigma)
    d = len(Z)
    Z = Z.reshape(1, d)
    #print(f"Z : {Z}")
    Zt = Z.T
    #print(f"Zt : {Zt}")
    a = d * np.log(2 * np.pi)
    #print(f"a = {a}")
    detsig = np.linalg.det(L)
    invsig = np.linalg.inv(L)
    b = 0.5 * np.log(detsig)
    #print(f"b = {b} : 0.5 * np.log(np.linalg.det(sigma)) with detsig = {detsig}")
    c = (0.5 * Z @ invsig @ Zt).flatten()[0]
    print(f"Pour lambda = {lbda}, Déterminant de sigma : {detsig} et resultat = {a+b+c}")
    return a + b + c


def logf(lambda_, x, z, n):
    lambda_ = float(lambda_) + 0.01 if lambda_ == 0 else lambda_
    # L_matrix = lu(covariance_matrix(x, lambda_))[1]
    # logdetS = slogdet(L_matrix.T)[1]
    # invS = L_matrix  # inv(L_matrix)
    sigma = sigmaKov(x, lambda_, kov)
    # L = lu(sigma)[1]
    logdetS = np.linalg.slogdet(sigma)[1]
    # logdetS = log(det(sigma))
    invS = np.linalg.inv(sigma)
    # U = lu(sigma)[1]
    # g = np.random.normal(size=x.shape[0])
    # mean = U @ g
    # mean = np.mean(x)
    # print(logdetS)
    return 0.5 * (n * np.log(2 * np.pi) + logdetS + z.T @ invS @ z)