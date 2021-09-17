import matplotlib.pyplot as plt
import numpy as np


def gen_data(dim_x, dim_y, dim_Z, times, seed=0):
    np.random.seed(seed)

    Z = []
    eps = []
    Y = []
    rho_x = 0.5
    eta = 1
    s = dim_Z
    X = []
    theta_row = np.array([[np.random.normal(3, 0.01) if i < s else 0 for i in range(dim_x)] for _ in range(dim_y)]).T
    A = np.array([np.random.multivariate_normal([0.5 * eta for _ in range(dim_x)], cov=np.array(
        [[0.01 * eta ** 2 if j == i else 0 for i in range(dim_x)] for j in range(dim_x)])) for _ in range(dim_Z)]).T
    b = np.array([np.random.multivariate_normal([0.1 for _ in range(dim_y)], cov=np.array(
        [[1 if j == i else 0 for i in range(dim_y)] for j in range(dim_y)])) for _ in range(dim_Z)])

    mat_proj_b = np.matmul(np.matmul(b.T, np.linalg.inv(np.matmul(b, b.T))), b)
    I_m = [[1 if i == j else 0 for i in range(dim_y)] for j in range(dim_y)]
    theta = np.matmul(theta_row, I_m - mat_proj_b)
    for t in range(1, times + 1):
        X += [list(np.random.multivariate_normal(np.array([0 for i in range(dim_x)]), cov=np.array(
            [[1 if j == i else rho_x * (-1) ** (j + i) for i in range(dim_x)] for j in range(dim_x)])))]
        W = np.random.multivariate_normal([0 for _ in range(dim_Z)], cov=np.array(
            [[1 if j == i else 0 for i in range(dim_Z)] for j in range(dim_Z)]))
        Z += [np.matmul(A.T, X[-1]) + W]
        eps += [list(np.random.multivariate_normal(np.array([0 for _ in range(dim_y)]), cov=np.array(
            [[1 if j == i else 0 for i in range(dim_y)] for j in range(dim_y)])))]
        Y += [np.matmul(theta.T, X[-1]) + np.matmul(b.T, Z[-1]) + eps[-1]]
    return X, Z, Y, theta, b


def hawkes_process(delta=0.1, lbda=0.5, kappa=0.5, times=10):
    T = 0
    i = 1
    arrival_times = []
    for i in range(times):
        lambda_star = lbda + kappa * np.exp(-delta * T)
        tau = -np.log(np.random.uniform(0, 1)) / lambda_star
        s = np.random.uniform(0, 1)
        T += tau
        if np.random.uniform(0,1) < lbda + kappa * np.exp(-delta * (T+ tau)):
            arrival_times += [T]
    return arrival_times

if __name__ == "__main__":
    """
    X, Y, B, rho, theta = gen_data(3, 2, 5, 10)
    print(rho)
    print(theta)
    plt.figure()
    plt.plot(B, color="red")
    plt.show()
    """
    times = hawkes_process(times=100)
    print(times)
    count_process = [0 for _ in range(len(times))]
    for j in range(len(times)):
        for time in times:
            if j < time and j+1 > time:
                count_process[j] += 1
    plt.figure()
    plt.scatter(times, count_process)
    plt.show()
