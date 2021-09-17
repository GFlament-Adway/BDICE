import numpy as np
from scipy.stats import invgauss, norm

def empirical_var(losses, alpha=0.01):
    sorted_liste = sorted(losses)
    if (len(losses)*alpha).is_integer():
        return sorted(losses)[int(len(losses)*alpha)]
    else:
        return (sorted_liste[int(len(losses)*alpha)] + sorted_liste[int(len(losses)*alpha) + 1])/2

def VaR(PD, rho, alpha=0.01, t=1):
    print(norm.ppf(alpha), norm.ppf(PD))
    return norm.cdf((norm.ppf(PD) - np.sqrt(rho)*norm.ppf(alpha)) / np.sqrt(1-rho), 0, scale=np.sqrt(t))


def VaR_mod(PD, rho, alpha=0.01, X=0, beta=0, t=1):
    return norm.cdf((norm.ppf(PD) - np.sqrt(rho)*norm.ppf(alpha, beta*X)) / np.sqrt(1-rho), 0, scale=np.sqrt(t))

def get_rho(X):
    return (np.matmul(X.T, X).sum() - np.matmul(X.T, X).trace())/(len(X)*len(X[0])*(len(X[0]) - 1))

def var_env(value, risks, alpha=0.01, treshold=None):
    if treshold is None:
        treshold = 0.9*value
    #return norm.ppf(alpha, 0, scale=np.sqrt(np.sum(risks))) + value
    return(norm.cdf(treshold - value, 0, scale= np.sum(risks)))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([VaR(0.001, 0.001*t, 0.004) for t in range(1000)])
    plt.xticks([100*t for t in range(11)], [np.round(0.1*t, 1) for t in range(11)])
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$PD_{stressee}(Y_T) (\%)$")
    plt.show()

