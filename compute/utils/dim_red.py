import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .utils_math import ls


def EM(data, q=1, energie=None):
    T = len(data[0])
    p = len(data)
    tau = np.zeros((p, p))
    np.fill_diagonal(tau, 1)
    id = np.zeros((q, q))
    np.fill_diagonal(id, 1)
    beta = np.array([[0.1 for _ in range(p)] for _ in range(q)])
    cyy = np.array([[np.mean([data[k][t] * data[i][t] for t in range(T)]) for i in range(p)] for k in range(p)])
    for _ in range(1000):
        delta = np.matmul(np.linalg.inv(tau + np.matmul(beta.T, beta)), beta.T)
        Delta = id - np.matmul(np.matmul(beta, np.linalg.inv(tau + np.matmul(beta.T, beta))), beta.T)
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(delta.T, cyy), delta) + Delta), delta.T), cyy.T)
        tau = cyy - np.matmul(
            np.matmul(np.matmul(cyy, delta), np.linalg.inv(np.matmul(np.matmul(delta.T, cyy), delta) + Delta)),
            np.matmul(cyy, delta).T)
        tau = np.diag(np.diag(tau))
    print("beta : ", beta)
    print("tau : ", tau)
    print("hat z : ", [np.matmul(delta.T, [data[k][t] for k in range(len(data))]) for t in range(T)])
    print("delta : ", delta)
    print("Delta : ", Delta)

    values = np.array(data).T
    z_hat = [np.matmul(values[t], delta) for t in range(values.shape[0])]
    print(np.array(z_hat).shape, beta.shape)
    plt.figure()
    for i in range(len(beta)):
        if i == 0:
            plt.plot([np.matmul(beta.T[i], z_hat[t]) for t in range(T)], color="red", label=r"$\Lambda_iY_t$")
        else:
            plt.plot([np.matmul(beta.T[i], z_hat[t]) for t in range(T)], color="red")
    if energie is not None:
        print(energie)
        plt.plot([k * 250 for k in range(36)], energie[0][-36:], color="black")
    for k, serie in enumerate(data):
        if k == 0:
            plt.plot(serie, color="blue", alpha=0.2, label=r"$X_{i,t}$")
        else:
            plt.plot(serie, color="blue", alpha=0.2)
    plt.xticks([k for k in range(1, T, 5*T // (30))], [2021 - 30 + 30 * k // T for k in range(1, T, 5*T // (30))],
               rotation=90)
    plt.legend()
    plt.draw()
    plt.figure()
    plt.plot([z[0] for z in z_hat])
    plt.draw()


def pca(values):
    plt.figure()

    pca = PCA(n_components="mle")
    principalComponents = pca.fit_transform(np.array(values).T)
    # plt.plot([value[0] for value in principalComponents], color="red")
    regs = []
    for serie in values:
        regs += [ls(serie, [value[0] for value in principalComponents], const=False, deg=1)]
    for reg in regs:
        plt.plot(reg["pred"], color="red")
    for value in values:
        plt.plot(value, color="blue", alpha=0.2)
    plt.draw()
