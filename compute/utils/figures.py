import matplotlib.pyplot as plt
import numpy as np


def illustre_metron(T, mu, sigma, D, n_sim=100):
    A = [[1] for _ in range(n_sim)]
    print(mu - sigma**2/2)
    for n in range(n_sim):
        for t in range(1, T):
            A[n] += [np.exp((mu - sigma ** 2 / 2) * t + sigma * np.random.normal(0, np.sqrt(t)))]

    plt.figure()
    for n in range(n_sim):
        plt.plot([k for k in range(1, T+1)], A[n], alpha=0.1, color="blue")
    plt.axhline(D, color="red")
    plt.show()


if __name__ == "__main__":
    illustre_metron(10, 0.03, 0.1, 0.8)