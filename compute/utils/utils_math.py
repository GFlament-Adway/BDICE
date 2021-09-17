import numpy as np
import matplotlib.pyplot as plt

def normalize(data, max=None):
    if max is None:
        max = np.max(data)
    return [value / max for value in data]


def cor(s1, s2):
    return np.corrcoef(s1, s2)[0, 1]

def diff(data):
    return([data[k]-data[k-1] for k in range(1, len(data))])

def accroissement(data):
    return([100*(data[k]-data[k-1])/data[k-1] for k in range(1, len(data))])

def ls(Y, X, deg=2, const=True, func="poly"):
    if func == "poly":
        if const:
            X = np.array([[value**i for i in range(deg+1)] for value in X])
        else:
            X = np.array([[value**i for i in range(1, deg+1)] for value in X])
    else:
        if const:
            X = np.array([[func(value**i) for i in range(deg+1)] for value in X])
        else:
            X = np.array([[func(value**i) for i in range(1, deg+1)] for value in X])
    Y = np.array(Y)
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)
    pred = [np.sum([beta[i]*X[t][i] for i in range(len(beta))]) for t in range(len(X))]
    epsilon = [(pred[t] - Y[t]) for t in range(len(pred))]
    result = {"Y": Y, "epsilon": epsilon, "pred": pred, "beta": beta, "eqm": np.mean([(pred[t] - Y[t])**2 for t in range(len(pred))])}
    return result

def mean_diff_length(data):
    n_years = max([len(serie) for serie in data])
    mean = [[] for _ in range(0, n_years + 1)]
    for year in range(n_years, 0, -1):
        for serie in data:
            if len(serie) - year >= 0:
                mean[year] += [serie[len(serie) - year]]
    mean = mean[::-1]
    mean = [value for value in mean if value != []]
    mean = [np.mean(value) for value in mean]
    return mean

def centre_red(values):
    for i, serie in enumerate(values):
        mean = np.mean(serie)
        sd = np.sqrt(np.var(serie))
        values[i] = [(serie[k] - mean)/sd for k in range(len(serie))]
    return values