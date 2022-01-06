import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats 



def alpha_estimator_moment(X):
    xbar2 = (1/X.shape[0]) * np.square(X).sum()
    return np.sqrt((xbar2 - np.square(X.mean())) / (np.square(np.pi)/6))


def m_estimator_moment(X):
    return X.mean() - np.euler_gamma * alpha_estimator_moment(X)


def estimators_moment(X):
    """Faster computation if the goal is to compute both the estimators
    """
    alpha = alpha_estimator_moment(X)
    m = X.mean() - np.euler_gamma * alpha
    return m, alpha


def alpha_estimator_likelihood(X):
    num = np.sum(X * np.exp(-X))
    den = np.sum(np.exp(-X))
    return X.mean() - num/den


def m_estimator_likelihood(X):
    alpha = alpha_estimator_likelihood(X)
    m = alpha * (np.log(len(X)) - np.log(np.sum(np.exp(-X/alpha))))
    return m


def estimators_likelihood(X):
    alpha = alpha_estimator_likelihood(X)
    m = alpha * (np.log(len(X)) - np.log(np.sum(np.exp(-X/alpha))))
    return m, alpha


def test_m_convergence(m, method, alpha=1, N = 1000):
    start = 5
    pas = 5
    # Moments
    if method == "moments":
        m_estimates = []
        for n in range(start, N, pas):
            samples = scipy.stats.gumbel_r.rvs(loc=m, scale=alpha, size=n)
            m_hat, alpha_hat = estimators_moment(samples)
            m_estimates.append(m_hat)
        fig, ax = plt.subplots()
        m_hat = np.asarray(m_estimates)[-20:].mean()

        ax.plot(range(start, N, pas), m_estimates, color="yellow", marker='.', linestyle="", label = r"$\hat{}$ = {}".format("m", m_hat.round(3)))
        mi, ma = plt.xlim()
        ax.hlines(m, mi, ma, color="red", linestyle="dashed", label="m = "+str(m), zorder=10)
        plt.grid()
        plt.title("Convergence de l'estimateur de m des moments")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()
    # Likelihood    
    if method == "likelihood":
        m_estimates = []
        for n in range(start, N, pas):
            samples = scipy.stats.gumbel_r.rvs(loc=m, scale=alpha, size=n)
            m_hat, alpha_hat = estimators_likelihood(samples)
            m_estimates.append(m_hat)
        fig, ax = plt.subplots()
        m_hat = np.asarray(m_estimates)[-20:].mean()

        ax.plot(range(start, N, pas), m_estimates, color="yellow", marker='.', linestyle="", label = r"$\hat{}$ = {}".format("m", m_hat.round(3)))
        mi, ma = plt.xlim()
        ax.hlines(m, mi, ma, color="red", linestyle="dashed", label="m = "+str(m), zorder=10)
        plt.grid()
        plt.title("Convergence de l'estimateur de m de la vraisemblance")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()



def test_alpha_convergence(alpha, method, m=0, N = 1000):
    start = 5
    pas = 5
    if method == "moments":
        alpha_estimates = []
        for n in range(start, N, pas):
            samples = scipy.stats.gumbel_r.rvs(loc=m, scale=alpha, size=n)
            m_hat, alpha_hat = estimators_moment(samples)
            alpha_estimates.append(alpha_hat)
        fig, ax = plt.subplots()
        alpha_hat = np.asarray(alpha_estimates)[-20:].mean()

        ax.plot(range(start, N, pas), alpha_estimates, color="blue", marker='x', linestyle="", label=r"$\hat{\alpha}$ = " + str(alpha_hat.round(3)))
        mi, ma = plt.xlim()
        ax.hlines(alpha, mi, ma, color="black", linestyle="dashed", label=r"$\alpha$ = " + str(alpha), zorder=10)
        plt.grid()
        plt.title(r"Convergence de l'estimateur de $\alpha$ des moments")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()
    if method == "likelihood":
        alpha_estimates = []
        for n in range(start, N, pas):
            samples = scipy.stats.gumbel_r.rvs(loc=m, scale=alpha, size=n)
            m_hat, alpha_hat = estimators_likelihood(samples)
            alpha_estimates.append(alpha_hat)
        fig, ax = plt.subplots()
        alpha_hat = np.asarray(alpha_estimates)[-20:].mean()

        ax.plot(range(start, N, pas), alpha_estimates, color="blue", marker='x', linestyle="", label=r"$\hat{\alpha}$ = " + str(alpha_hat.round(3)))
        mi, ma = plt.xlim()
        ax.hlines(alpha, mi, ma, color="black", linestyle="dashed", label=r"$\alpha$ = " + str(alpha), zorder=10)
        plt.grid()
        plt.title(r"Convergence de l'estimateur de $\alpha$ de la vraisemblance")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()