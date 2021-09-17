import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

def est_f(Y, X, lambda_1, lambda_2):
    if type(X) != np.ndarray and type(X) == list:
        X = np.array(X)
    if type(Y) != np.ndarray and type(Y) == list:
        Y = np.array(Y)
    n = X.shape[0]
    p = X.shape[1]
    m = Y.shape[1]
    id_n = [[1 if j == i else 0 for i in range(n)] for j in range(n)]
    id_lambda_1 = [[lambda_1 if j == i else 0 for i in range(p)] for j in range(p)]
    n_lambda_2 = [[X.shape[0] * lambda_2 if j == i else 0 for i in range(p)] for j in range(p)]
    P_lambda_2 = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(X.T, X) + n_lambda_2)), X.T)
    Q_lambda_2 = id_n - P_lambda_2
    sqrt_q = sqrtm(Q_lambda_2)
    proj_X = np.matmul(X.T, sqrt_q)
    theta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(proj_X, X) + id_lambda_1), proj_X), Y)
    l_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X) + n_lambda_2), X.T), Y - np.matmul(X, theta_hat))
    return np.array([[float(theta_hat[j][i].real + l_hat[j][i].real) for i in range(m)] for j in range(p)])


def est_pb(Y, X, f, K):
    if type(X) != np.ndarray and type(X) == list:
        X = np.array(X)
    if type(Y) != np.ndarray and type(Y) == list:
        Y = np.array(Y)
    y_hat = np.matmul(X, f)
    errors = np.array([Y[i] - y_hat[i] for i in range(len(Y))])
    sigma_errors = 1 / X.shape[0] * np.matmul(errors.T, errors)
    evalues, evectors = np.linalg.eig(sigma_errors)
    K_evectors = [list(evalues).index(np.sort(evalues)[-(i+1)]) for i in range(K)]
    U = np.array([evectors[vect_indices] for vect_indices in K_evectors])
    P_b = np.matmul(U.T, U)
    return P_b


def est_theta(Y, X, P_b, lambda_3):
    if type(X) != np.ndarray and type(X) == list:
        X = np.array(X)
    if type(Y) != np.ndarray and type(Y) == list:
        Y = np.array(Y)

    m = Y.shape[1]
    p = X.shape[1]
    lambda_p = [[lambda_3 if j == i else 0 for i in range(p)] for j in range(p)]
    I_m = [[1 if j == i else 0 for i in range(m)] for j in range(m)]
    inv = np.linalg.inv(np.matmul(X.T, X) + lambda_p)
    proj_y = np.matmul(X.T, np.matmul(Y, I_m - P_b))
    theta_hat = np.matmul(inv, proj_y)
    return [[theta_hat[j][i].real for i in range(m)] for j in range(p)]


def ridge_reg(Y,X, lambda_1):
    if type(X) != np.ndarray and type(X) == list:
        X = np.array(X)
    if type(Y) != np.ndarray and type(Y) == list:
        Y = np.array(Y)
    p = X.shape[1]
    i_lmbda = [[lambda_1 if j==i else 0 for j in range(p)] for i in range(p)]
    proj = np.matmul(X.T,X) + i_lmbda
    inv = np.linalg.inv(proj)
    return np.matmul(np.matmul(inv, X.T), Y)

if __name__ == "__main__":
    from gen_data import gen_data

    MSE_first_steps = []
    MSE_ridges = []
    MSE_finals = []

    K=3
    p=150
    m=20
    n=100
    for k in range(10):
        X, Z, Y, theta, b = gen_data(p, m, K, n, seed=k)
        #print("true theta : ", theta)
        P_b = np.matmul(np.matmul(b.T, np.linalg.inv(np.matmul(b, b.T))), b)

        f = est_f(Y, X, 0.2, 20)
        #print("first theta : ", f)
        y_hat_first_step = np.matmul(X, f).T
        P_b_est = est_pb(Y, X, f, K)
        #print(P_b)
        theta_hat = est_theta(Y, X, P_b_est, 1)
        #print("theta hat : ", theta_hat)
        theta_hat_ridge = ridge_reg(Y, X, 0.2)
        #print("theta ridge : ", theta_hat_ridge)
        y_hat_ridge = np.matmul(X, theta_hat_ridge).T
        y_hat = np.matmul(X, theta_hat).T
        Y = np.array(Y).T

        MSe_ridge = [np.mean([(theta[i][t] - theta_hat_ridge[i][t]) ** 2 for t in range(m)]) for i in range(p)]
        MSE_first_step = [np.mean([(theta[i][t] - f[i][t]) ** 2 for t in range(m)]) for i in range(p)]
        MSE_first_steps += [np.mean(MSE_first_step)]
        MSE_ridges += [np.mean(MSe_ridge)]
        MSE = [np.mean([(theta[i][t] - theta_hat[i][t]) ** 2 for t in range(m)]) for i in range(p)]
        MSE_finals += [np.mean(MSE)]
    print("ridge : ", np.mean(MSE_ridges))
    print("MSE first : ", np.mean(MSE_first_steps))
    print("MSE : ", np.mean(MSE_finals))

    """
    plt.figure()

    colors = ["blue", "red", "green", "brown", "black", "magenta", "cyan"]
    for i in [1]:
        plt.plot(y_hat[i], color=colors[i], alpha=0.5, marker="*", label="Prédiction")
        plt.plot(Y[i], color=colors[i-1], marker="o", alpha=0.5, label="Observation")
        plt.plot(y_hat_ridge[i], color=colors[i], alpha=0.5, label="Prédiction ridge")
    plt.legend()
    plt.show()
    """
