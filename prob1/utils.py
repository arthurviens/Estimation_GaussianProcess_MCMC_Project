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


def f(alpha, Xi):
    m = np.mean(Xi)
    num = np.sum(Xi * np.exp(-Xi/alpha))
    den = np.sum(np.exp(-Xi/alpha))
    return alpha - m + num/den

def f_prime(alpha, Xi):
    term1 = np.sum(np.exp(-Xi/alpha)) 
    term2 = np.sum(np.square(Xi/alpha) * np.exp(-Xi/alpha))
    term3 = np.sum(Xi * np.exp(-Xi/alpha))
    term4 = np.sum(Xi/alpha**2 * np.exp(-Xi/alpha))
    term5 = np.square(np.sum(np.exp(-Xi/alpha)))
    return 1 + (term1 * term2 - term3 * term4) / term5 

def newton(f,Df,x0, X, epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn, X)
        if abs(fxn) < epsilon:
            #print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn, X)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None



def alpha_estimator_likelihood(X):
    return newton(f, f_prime, 2, X, 1e-7, 100)


def m_estimator_likelihood(X):
    alpha = alpha_estimator_likelihood(X)
    m = alpha * (np.log(len(X)) - np.log(np.sum(np.exp(-X/alpha))))
    return m


def estimators_likelihood(X):
    alpha = alpha_estimator_likelihood(X)
    m = alpha * (np.log(len(X)) - np.log(np.sum(np.exp(-X/alpha))))
    return m, alpha

def test_m_convergence(m, method, alpha=1, N = 1000):
    start = 20
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

        ax.plot(range(start, N, pas), m_estimates, color="green", marker='o', linestyle="", \
            label = r"$\hat{}$ = {}".format("m", m_hat.round(3)), ms=100./fig.dpi)
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

        ax.plot(range(start, N, pas), m_estimates, color="yellow", marker='o', linestyle="", \
            label = r"$\hat{}$ = {}".format("m", m_hat.round(3)), ms=100./fig.dpi)
        mi, ma = plt.xlim()
        ax.hlines(m, mi, ma, color="red", linestyle="dashed", label="m = "+str(m), zorder=10)
        plt.grid()
        plt.title("Convergence de l'estimateur de m de la vraisemblance")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()
    if method == "both":
        m_estimates_moment = []
        m_estimates_likeli = []
        for n in range(start, N, pas):
            samples = scipy.stats.gumbel_r.rvs(loc=m, scale=alpha, size=n)
            m_hat_m, alpha_hat = estimators_moment(samples)
            m_hat_l, alpha_hat = estimators_likelihood(samples)
            m_estimates_moment.append(m_hat_m)
            m_estimates_likeli.append(m_hat_l)
        fig, ax = plt.subplots()
        m_hat_m = np.asarray(m_estimates_moment)[-20:].mean()
        m_hat_l = np.asarray(m_estimates_likeli)[-20:].mean()
        
        ax.plot(range(start, N, pas), m_estimates_moment, color="green", marker='o', ms=60./fig.dpi, \
            linestyle="", label = r"$\hat{}$ moment = {}".format("m", m_hat_m.round(3)))
        ax.plot(range(start, N, pas), m_estimates_likeli, color="yellow", marker='o', ms=60./fig.dpi, \
            linestyle="", label = r"$\hat{}$ likeli = {}".format("m", m_hat_l.round(3)))
        mi, ma = plt.xlim()
        ax.hlines(m, mi, ma, color="red", linestyle="dashed", label="m = "+str(m), zorder=10)
        plt.grid()
        plt.title("Convergence de l'estimateur de m des moments et vraisemblance")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()

def test_alpha_convergence(alpha, method, m=0, N = 1000):
    start = 20
    pas = 5
    if method == "moments":
        alpha_estimates = []
        for n in range(start, N, pas):
            samples = scipy.stats.gumbel_r.rvs(loc=m, scale=alpha, size=n)
            m_hat, alpha_hat = estimators_moment(samples)
            alpha_estimates.append(alpha_hat)
        fig, ax = plt.subplots()
        alpha_hat = np.asarray(alpha_estimates)[-20:].mean()

        ax.plot(range(start, N, pas), alpha_estimates, color="green", marker='x', linestyle="", \
            label=r"$\hat{\alpha}$ = " + str(alpha_hat.round(3)), ms=100./fig.dpi)
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

        ax.plot(range(start, N, pas), alpha_estimates, color="yellow", marker='x', linestyle="", \
            label=r"$\hat{\alpha}$ = " + str(alpha_hat.round(3)), ms=100./fig.dpi)
        mi, ma = plt.xlim()
        ax.hlines(alpha, mi, ma, color="black", linestyle="dashed", label=r"$\alpha$ = " + str(alpha), zorder=10)
        plt.grid()
        plt.title(r"Convergence de l'estimateur de $\alpha$ de la vraisemblance")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()
    if method == "both":
        alpha_estimates_moment = []
        alpha_estimates_likeli = []
        for n in range(start, N, pas):
            samples = scipy.stats.gumbel_r.rvs(loc=m, scale=alpha, size=n)
            m_hat_m, alpha_hat_m = estimators_moment(samples)
            m_hat_l, alpha_hat_l = estimators_likelihood(samples)
            alpha_estimates_moment.append(alpha_hat_m)
            alpha_estimates_likeli.append(alpha_hat_l)
        fig, ax = plt.subplots()
        alpha_hat_m = np.asarray(alpha_estimates_moment)[-20:].mean()
        alpha_hat_l = np.asarray(alpha_estimates_likeli)[-20:].mean()

        ax.plot(range(start, N, pas), alpha_estimates_moment, color="green", marker='o', ms=60./fig.dpi, \
            linestyle="", label = r"$\hat{}$ moment = {}".format(r"\alpha", alpha_hat_m.round(3)))
        ax.plot(range(start, N, pas), alpha_estimates_likeli, color="yellow", marker='o', ms=60./fig.dpi, \
            linestyle="", label = r"$\hat{}$ likeli = {}".format(r"\alpha", alpha_hat_l.round(3)))
        mi, ma = plt.xlim()
        ax.hlines(alpha, mi, ma, color="black", linestyle="dashed", label=r"$\alpha$ = "+str(alpha), zorder=10)
        plt.grid()
        plt.title(r"Convergence de l'estimateur de $\alpha$ des moments et vraisemblance")
        plt.xlabel("Taille de l'échantillon")
        plt.ylabel("Valeur du paramètre")
        plt.legend()
        plt.show()