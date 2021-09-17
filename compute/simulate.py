import numpy as np


def simulate_loss(mean=0, sd=1, size=10000):
    losses = np.random.normal(mean, sd, size)
    return [np.float(loss) for loss in losses]


def simulate_PD(p=0.01, size=100):
    return np.random.binomial(1, p, size)

def simulate_values(cov=None, mean=None, time_horizon=100):
    assert not isinstance(mean, (str, type(None))), "Il faut fournir un vecteur d'espérances"
    if cov is None:
        cov = [[1 if k == j else 0 for k in range(len(mean))] for j in range(len(mean))]
    return np.random.multivariate_normal(mean, cov, size=time_horizon)

def simulate_rho(rho, n_company):
    values = simulate_values([[rho if j != i else 1 for j in range(n_company)] for i in range(n_company)], [0 for _ in range(n_company)])
    return values

if __name__ == "__main__":
    values = simulate_rho(0.2, 100)
    print(np.array(values).shape)
    pd = simulate_PD()
    print(pd)
