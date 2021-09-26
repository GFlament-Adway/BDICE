import pymc3 as pm
import arviz as az
import numpy as np
from utils.utils import tfp, countries_energy, diff, get_countries
import matplotlib.pyplot as plt
import pandas as pd
import theano


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


if __name__ == "__main__":
    print(pm.__version__)
    print(az.__version__)

    countries = get_countries("data/iea/tfp_countries.csv")[:-2]
    countries = ["United States", "Japan", "Italy", "Germany", "United Kingdom", "France", "Canada"]

    starting_year = 1971
    end_year = 2019
    y = np.array(
        [diff(tfp("data/iea/tfp_countries.csv", country=country)[country][1:-1]) for country in countries])
    x = [countries_energy(country=country, starting_year=starting_year, end_year=end_year) for country in countries]
    x = [[x[k]["country"][country][starting_year + year + 1]["exergy"]
          for year in range(end_year - starting_year - 1)] for k, country in enumerate(countries)]
    x = np.array([diff([v / x[k][0] for v in x[k]]) for k, country in enumerate(countries)])
    country_id = np.array([[k for _ in range(x.shape[1])] for k in range(len(countries))]).flatten()
    x_exergy = np.array(
        [diff(tfp("data/iea/tfp_countries.csv", country=country)[country][:-2]) for country in countries])

    x = x.flatten()
    y = y.flatten()
    x_exergy = x_exergy.flatten()

    alpha = 0.05

    with pm.Model() as model:
        # Define priors

        sigma = pm.HalfCauchy("sigma", beta=10, shape=len(countries))
        intercept = pm.Normal("Intercept", 0, sigma=0.1, shape=len(countries))

        mu_x = pm.Normal("prior exergy coefficient", 0.1, sigma=0.1)
        sigma_x_coef = pm.HalfCauchy("sigma_exergy_coef", beta=5)

        mu_ar = pm.Normal("priori ar", 0, sigma=0.1)
        sigma_x_ar = pm.HalfCauchy("sigma_ar_coef", beta=5)

        x_coeff = pm.Normal("exergy coefficient", mu_x, sigma=sigma_x_coef, shape=len(countries))
        x_ar = pm.Normal("ar", mu_ar, sigma=sigma_x_ar, shape=len(countries))

        # Define likelihood
        likelihood = pm.Normal("y",
                               mu=intercept[country_id] + x_coeff[country_id] * x + x_ar[country_id] * x_exergy,
                               sigma=sigma[country_id],
                               observed=y)
        # Inference!
        trace = pm.sample(draws=100, tune=20000, cores=2)

    with model:
        predictions = pm.sample_posterior_predictive(trace, var_names=["Intercept", "exergy coefficient", 'ar', "y"],
                                                     random_seed=5)

        az.plot_trace(trace, var_names=['sigma', 'exergy coefficient', 'ar'], legend=True)
        idata = az.from_pymc3(model=model, posterior_predictive=predictions)
        az.plot_ppc(idata)
        mu_p = predictions["y"]
        print(mu_p.shape)
        lower = [np.sort(mu_p.T[k])[int(len(mu_p.T[k])*alpha)] for k in range(len(mu_p[0]))]
        upper = [np.sort(mu_p.T[k])[int(len(mu_p.T[k]) * (1-alpha))] for k in range(len(mu_p[0]))]
        print("#################")
        print(mse(mu_p.mean(0), y))
        print(r2(mu_p.mean(0), y))
        print("#################")
        fig, ax = plt.subplots()
        ax.plot(mu_p.mean(0), label="Mean TFP prediction", alpha=0.6)
        ax.plot(y, ms=4, alpha=0.4, label="Data")
        ax.fill_between([k for k in range(len(mu_p[0]))], lower, upper, alpha=0.1)
        ax.legend()
        plt.show()
