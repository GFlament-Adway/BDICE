import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde, norm

import VaR
import simulate


def compute_var_norm():
    alphas = [0.9, 0.99]
    plt.figure()
    ns = [250, 500, 1000]
    for alpha in alphas:
        for n in ns:
            var = norm.ppf(alpha)
            var_est = np.sqrt(alpha * (1 - alpha)) / (np.sqrt(n) * norm.pdf(norm.ppf(alpha)))
            vars = np.random.normal(var, var_est, size=10000)
            xs = np.linspace(min(vars), max(vars), 1000)
            density = gaussian_kde(vars)
            plt.plot(xs, [density(x) for x in xs], label=r"n = {n} et $\alpha$ = {alpha}".format(n=n, alpha=alpha))
    plt.legend()
    plt.draw()


def est_var_norm():
    alphas = [0.9, 0.99]
    ns = [250, 500, 1000]

    plt.figure()
    for alpha in alphas:
        for n in ns:
            losses = simulate.simulate_loss(0, 1, n)
            density = gaussian_kde(losses)
            xss = [np.linspace(-20, x, 100) for x in np.linspace(-1, 5, 100)]
            cdf = [np.sum([(density(xs[k + 1]) + density(xs[k])) / 2 * (xs[k + 1] - xs[k]) for k in range(len(xs) - 1)])
                   for xs
                   in xss]
            var = VaR.empirical_var(losses, alpha=alpha)
            print(var)
            print(cdf)
            ppf = xss[next(x[0] for x in enumerate(cdf) if x[1] > alpha)][-1]
            print(ppf)
            var_est = np.sqrt(alpha * (1 - alpha)) / (np.sqrt(n) * density(ppf))

            vars = np.random.normal(var, var_est, size=10000)
            xs = np.linspace(min(vars), max(vars), 1000)
            density = gaussian_kde(vars)
            plt.plot(xs, [density(x) for x in xs], label=r"n = {n} et $\alpha$ = {alpha}".format(n=n, alpha=alpha))
    plt.legend()
    plt.show()

def dvar():
    rhos = [x for x in np.linspace(0, 0.9, 100)]
    PDs = [pd for pd in np.linspace(0.01, 0.9, 100)]
    vars = []
    for PD in PDs:
        vars_pd = []
        for rho in rhos:
            #PD = simulate.simulate_PD()
            #values = simulate.simulate_values(mean=[0, 0, 0], cov=[[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])
            #rho_chap = VaR.rho(values)
            vars_pd += [VaR.VaR(PD, rho)]
        vars += [vars_pd]
    X, Y = np.meshgrid(rhos, PDs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, np.array(vars))
    plt.show()

def ineg_bernstein():

    plt.figure()
    gauss = np.random.normal(0, 1, 100)
    sum_gauss = [np.sum(gauss[:k]) for k in range(len(gauss))]
    plt.plot(sum_gauss, label=r"Somme des $X_{i,t}$", alpha=0.05, color="blue")
    for _ in range(1000):
        gauss = np.random.normal(0, 1, len(gauss))
        sum_gauss = [np.sum(gauss[:k]) for k in range(len(gauss))]
        plt.plot(sum_gauss, alpha=0.05, color="blue")

    gamma = np.mean([(v - np.mean(gauss)) ** 2 for v in gauss])
    print(gamma)
    sup = [np.sqrt(-2 * gamma * k * np.log(0.1/2)) for k in range(len(gauss))]
    inf = [-np.sqrt(-2 * gamma * k * np.log(0.1/2)) for k in range(len(gauss))]
    plt.plot(inf, color="red", label=r"Borne inférieure pour $\alpha=0.1$")
    plt.plot(sup, color="red", label=r"Borne supérieure pour $\alpha=0.1$")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ineg_bernstein()