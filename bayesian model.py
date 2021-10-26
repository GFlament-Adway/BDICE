import pymc3 as pm
import arviz as az
import numpy as np
from utils.utils import med_regression
from utils.utils_bayes import bayesian_model, get_data
import matplotlib.pyplot as plt
import csv
import matplotlib
from DICE.pydice import DICE
import scipy.stats as stats


def from_posterior(param, samples, k=100):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, k)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)

def mse(y_hat, y):
    """

    :param y_hat: predicted values
    :param y: observed values
    :return: mean squared error
    """
    return np.mean([(y_hat[k] - y[k]) ** 2 for k in range(len(y))])


def r2(y_hat, y):
    """

    :param y_hat: predicted values
    :param y: observed values
    :return: R^2
    """
    sum_var = np.sum([(y[k] - np.mean(y)) ** 2 for k in range(len(y))])
    return 1 - np.sum([((y_hat[k] - y[k]) ** 2) for k in range(len(y_hat))]) / sum_var


def mpe(y_hat, y):
    """
    :param y_hat: predicted values
    :param y: observed values
    :return: mean percentage error
    """
    return 100 * np.mean([np.abs((y_hat[k] - y[k]) / y[k]) for k in range(len(y))])


def predict(traj_e, country="United States", path="data/posterior_predictive.csv", horizon=30, sample_size=1000,
            delta_a=-0.02):
    predictions = []
    with open(path, "r") as csv_file:
        names = csv.reader(csv_file)
        for i, data in enumerate(names):
            if i == 0:
                name = data
                coefs = {j: [] for j in name}
            else:
                for j in name:
                    coefs[j] += [data[name.index(j)]]
    sample_alpha_0 = [np.random.choice([k for k in range(len(coefs["alpha_0"]))]) for j in range(sample_size)]
    sample_alpha_1 = [np.random.choice([k for k in range(len(coefs["alpha_0"]))]) for j in range(sample_size)]
    sample_beta_0 = [np.random.choice([k for k in range(len(coefs["alpha_0"]))]) for j in range(sample_size)]

    predictions += [[np.random.normal(
        loc=np.sum(
            [float(coefs["alpha_0"][sample_alpha_0[j]]) * float(coefs["alpha_1"][sample_alpha_1[j]]) ** i for i in range(t)]) +
            float(coefs["alpha_1"][sample_alpha_1[j]]) ** t * delta_a +
            np.sum([float(coefs["beta_0"][sample_beta_0[j]]) * traj_e[i] * float(coefs["alpha_1"][sample_alpha_1[j]]) ** (i) for i in range(t)]),
        scale=np.sum([float(coefs["alpha_1"][sample_alpha_1[j]]) ** (2 * i) * np.mean([float(s) for s in coefs["sigma"]]) for i in range(t)]))
                    for t in range(1, horizon + 1)] for j in range(sample_size)]

    """
    
    """
    return predictions


def reg_median(x, y):
    beta_25, alpha_25 = med_regression(x, y, n_deciles=2, alpha=0.25)
    beta_5, alpha_5 = med_regression(x, y, n_deciles=2, alpha=0.5)
    beta_75, alpha_75 = med_regression(x, y, n_deciles=2, alpha=0.75)
    print(beta_25, beta_5, beta_75)
    plt.figure()
    plt.scatter(x, y, label="observations", alpha=0.5)
    plt.plot(x, [alpha_25 + beta_25 * x_value for x_value in x], color="red", alpha=0.5)
    plt.plot(x, [alpha_5 + beta_5 * x_value for x_value in x], label="regression quantile", color="red")
    plt.plot(x, [alpha_75 + beta_75 * x_value for x_value in x], color="red", alpha=0.5)
    plt.legend()
    plt.ylabel("log tfp variation")
    plt.xlabel("log exergy variation")
    plt.show()


def visu(predictions, countries, idata, y):
    az.plot_ppc(idata)
    mu_p = predictions["y"]
    lower = [np.sort(mu_p.T[k])[int(len(mu_p.T[k]) * alpha)] for k in range(len(mu_p[0]))]
    upper = [np.sort(mu_p.T[k])[int(len(mu_p.T[k]) * (1 - alpha))] for k in range(len(mu_p[0]))]
    print("#################")
    print(mse(mu_p.mean(0), y))
    print(r2(mu_p.mean(0), y))
    print("#################")
    fig, ax = plt.subplots()
    ax.plot(mu_p.mean(0), label="Prédictions faites à partir de l'énergie consommée", alpha=0.6)
    ax.plot(y, ms=4, alpha=0.4, label="Variations historiques de la productivité des facteurs")
    ax.fill_between([k for k in range(len(mu_p[0]))], lower, upper, alpha=0.1)
    ax.vlines([(k) * 46 for k in range(len(countries) + 1)], ymin=min(y), ymax=max(y), color="red")
    for k in range(len(countries)):
        ax.text(x=(k + 0.5) * 46 - len(countries[k]), y=max(y), s=countries[k])
    ax.legend(loc="lower center")
    ax.set_ylabel("Variations (%)")
    plt.show()


if __name__ == "__main__":
    print(pm.__version__)
    print(az.__version__)
    plt.style.use('ggplot')
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)


    horizon = 30
    alpha=0.1
    preds = np.array(predict([np.log(1 - 0.04) for _ in range(horizon)], country="United States", sample_size=500,
                    horizon=horizon, delta_a=0)).T

    preds_g = np.array(predict([np.log(1 + 0.04) for _ in range(horizon)], country="United States", sample_size=500,
                    horizon=horizon, delta_a=0)).T


    q_5 = [np.sort(preds[t])[int(len(preds[t]) * alpha)] for t in range(horizon)]
    q_95 = [np.sort(preds[t])[int(len(preds[t]) * (1-alpha))] for t in range(horizon)]

    plt.figure()
    plt.plot(preds, color="blue", alpha=0.01)
    plt.plot(q_5, color="red", alpha=0.5, linestyle=":", label="Quantiles of the predictions")
    plt.plot(q_95, color="red", alpha=0.5)
    plt.plot([np.median(preds[t]) for t in range(len(preds))], color="red", label="Median values of the predictions")
    plt.xlabel("Years of forecast")
    plt.ylabel("log variations of the TFP")
    plt.legend(loc="upper right")
    plt.draw()


    q_g_5 = [np.sort(preds_g[t])[int(len(preds[t]) * alpha)] for t in range(horizon)]
    q_g_95 = [np.sort(preds_g[t])[int(len(preds[t]) * (1-alpha))] for t in range(horizon)]

    plt.figure()
    plt.plot(preds_g, color="blue", alpha=0.01)
    plt.plot(q_g_5, color="red", alpha=0.5, linestyle=":", label="Quantiles of the predictions")
    plt.plot(q_g_95, color="red", alpha=0.5)
    plt.plot([np.median(preds_g[t]) for t in range(len(preds))], color="red", label="Median values of the predictions")
    plt.xlabel("Years of forecast")
    plt.ylabel("log variations of the TFP")
    plt.legend(loc="upper right")
    plt.draw()


    dice_q_5 = DICE(tfp=q_5)
    dice_q_95 = DICE(tfp=q_95)
    dice_med = DICE(tfp=[np.median(preds[t]) for t in range(horizon)])
    dice_med_growth = DICE(tfp=[np.median(preds_g[t]) for t in range(horizon)])
    dice_med_growth_q_5 = DICE(tfp=[np.sort(preds_g[t])[int(alpha*len(preds_g[t]))] for t in range(horizon)])
    dice_med_growth_q_95 = DICE(tfp=[np.sort(preds_g[t])[int((1-alpha)*len(preds_g[t]))] for t in range(horizon)])
    dice = DICE()
    for _ in range(len(q_5)):
        dice_q_5.step()
        dice_q_95.step()
        dice_med.step()
        dice_med_growth.step()
        dice.step()
        dice_med_growth_q_5.step()
        dice_med_growth_q_95.step()

    plt.figure()
    plt.title("TFP variations")
    plt.plot(dice.parameters["tfp"], color="blue", label="Nordhaus assumptions")
    plt.plot(dice_med_growth.parameters["tfp"], color="red", alpha=1, label="Median TFP value with exergy growth")
    plt.plot(dice_med_growth_q_95.parameters["tfp"], color="red", alpha=0.5, linestyle=":", label="Quantile with exergy growth")
    plt.plot(dice_med_growth_q_5.parameters["tfp"], color="red", alpha=0.5, linestyle=":")
    plt.plot(dice_q_5.parameters["tfp"], color="green", alpha=0.5, linestyle = ":", label="Quantile with exergy decline")
    plt.plot(dice_q_95.parameters["tfp"], color="green", alpha=0.5, linestyle=":")
    plt.plot(dice_med.parameters["tfp"], color="green", alpha=1, label="Median TFP value with exergy decline")
    plt.xlabel("Years of prediction")
    plt.legend()
    plt.draw()
    

    plt.figure()
    plt.title("Economic output")
    plt.plot(dice.parameters["output"], color="blue", label="Nordhaus assumptions")
    plt.plot(dice_med_growth.parameters["output"], color="red", alpha=1, label="Output with exergy growth")
    plt.plot(dice_med_growth_q_95.parameters["output"], color="red", alpha=0.5, linestyle=":", label="Quantile with exergy growth")
    plt.plot(dice_med_growth_q_5.parameters["output"], color="red", alpha=0.5, linestyle=":")
    plt.plot(dice_q_5.parameters["output"], color="green", alpha=0.5, linestyle=":", label="Quantile with exergy decline")
    plt.plot(dice_q_95.parameters["output"],color="green", alpha=0.5, linestyle=":")
    plt.plot(dice_med.parameters["output"], color="green", alpha=1, label="Median output value with exergy decline")
    plt.xlabel("Years of prediction")
    plt.ylabel("Trillions of $2010")
    plt.legend()
    plt.draw()

    data = get_data()
    predictions, countries, idata, y = bayesian_model(data=data)
    visu(predictions, countries, idata, y)
    plt.show()