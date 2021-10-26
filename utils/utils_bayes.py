import pymc3 as pm
import numpy as np
import arviz as az
from utils.utils import get_countries, diff, countries_energy, tfp
import csv
import scipy.stats as stats

def register(idata, country_id, country_name):
    alpha_0 = [idata["posterior_predictive"][r"$\alpha_0$"][0][t][country_id] for t in
               range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    alpha_1 = [idata["posterior_predictive"][r"$\alpha_1$"][0][t][country_id] for t in
               range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    beta_0 = [idata["posterior_predictive"][r"$\beta_0$"][0][t][country_id] for t in
              range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    beta_1 = [idata["posterior_predictive"][r"$\beta_1$"][0][t][country_id] for t in
              range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    sigma = [idata["posterior_predictive"]["sigma"][0][t][country_id] for t in
             range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    names = ["alpha_0", "alpha_1", "beta_0", "beta_1", "sigma"]
    rows = np.array([alpha_0, alpha_1, beta_0, beta_1, sigma]).T

    with open("data/{country}/posterior_predictive.csv".format(country=country_name), "w", newline="") as csv_file:
        write = csv.writer(csv_file)
        write.writerow(names)
        for row in rows:
            write.writerow(row)
def get_data(countries = None):
    """

    :param countries:
    :return:
    """

    if countries is None:
        countries = ["United States", "Japan", "Italy", "Germany", "United Kingdom", "France", "Canada"]

    starting_year = 1971
    end_year = 2019
    y = np.array(
        [diff(np.log(tfp("data/iea/tfp_countries.csv", country=country)[country][1:-1])) for country in countries])
    x = [countries_energy(country=country, starting_year=starting_year, end_year=end_year) for country in countries]
    # x = [countries_energy(country=country, starting_year=starting_year, end_year=end_year) for country in countries]
    x = [[np.log(x[k]["country"][country][starting_year + year + 1]["exergy"])
          for year in range(end_year - starting_year - 1)] for k, country in enumerate(countries)]
    # x = [[np.log(np.sum([x[k]["country"][country][starting_year + year + 1][source] for source in sources]) )
    #      for year in range(end_year - starting_year - 1)] for k, country in enumerate(countries)]

    x = np.array([diff([v for v in x[k]]) for k, country in enumerate(countries)])
    # x = np.array([[v / x[k][0] for v in x[k]] for k, country in enumerate(countries)])

    x_us = countries_energy(country="United States", starting_year=starting_year, end_year=end_year)
    x_ge = countries_energy(country="Germany", starting_year=starting_year, end_year=end_year)
    x_fr = countries_energy(country="France", starting_year=starting_year, end_year=end_year)
    x_jp = countries_energy(country="Japan", starting_year=starting_year, end_year=end_year)
    x_uk = countries_energy(country="United Kingdom", starting_year=starting_year, end_year=end_year)
    x_it = countries_energy(country="Italy", starting_year=starting_year, end_year=end_year)
    x_ca = countries_energy(country="Canada", starting_year=starting_year, end_year=end_year)
    x_us = [[np.log(x_us["country"]["United States"][starting_year + year + 1]["exergy"]
                    + x_ge["country"]["Germany"][starting_year + year + 1]["exergy"]
                    + x_fr["country"]["France"][starting_year + year + 1]["exergy"]
                    + x_jp["country"]["Japan"][starting_year + year + 1]["exergy"]
                    + x_uk["country"]["United Kingdom"][starting_year + year + 1]["exergy"]
                    + x_it["country"]["Italy"][starting_year + year + 1]["exergy"]
                    + x_ca["country"]["Canada"][starting_year + year + 1]["exergy"])
             for year in range(end_year - starting_year - 1)] for k, country in enumerate(countries)]
    x_us = np.array([diff([v for v in x_us[k]]) for k, country in enumerate(countries)])
    # x_us = np.array([[v / x_us[k][0] for v in x_us[k]] for k, country in enumerate(countries)])
    country_id = np.array([[k for _ in range(x.shape[1])] for k in range(len(countries))]).flatten()
    x_exergy = np.array(
        [diff(tfp("data/iea/tfp_countries.csv", country=country)[country][:-2]) for country in countries])
    # x_exergy = np.array(
    #    [tfp("data/iea/tfp_countries.csv", country=country)[country][:-2] for country in countries])
    x = x.flatten()
    y = y.flatten()

    x_us = x_us.flatten()
    x_exergy = x_exergy.flatten()

    alpha = 0.1
    data = {}
    data["countries"], data["country_id"], data["x"], data["x_exergy"], data["x_us"], data["y"] = countries, country_id, x, x_exergy, x_us, y
    return data

def bayesian_model(data, register_data = True):
    """

    :param register_data:
    :return:
    """
    countries, country_id, x, x_exergy, x_us, y = data["countries"], data["country_id"], data["x"], data["x_exergy"], data["x_us"], data["y"]
    with pm.Model() as model:
        # Define priors
        sigma = pm.HalfCauchy("sigma", beta=1, shape=len(countries))
        intercept = pm.Normal(r"$\alpha_0$", 0, sigma=1, shape=len(countries))

        x_offset = pm.Normal('x_offset', mu=0, sd=1, shape=len(countries))
        mu_x = pm.Normal("prior exergy coefficient", 1, sigma=1)
        sigma_x_coef = pm.HalfCauchy("sigma_exergy_coef", beta=10)

        ar_offset = pm.Normal('ar_offset', mu=0, sd=1, shape=len(countries))
        mu_ar = pm.Normal(r"$\mu_{\alpha_1}$", 0, sigma=1)
        sigma_x_ar = pm.HalfCauchy("sigma_ar_coef", beta=10)

        mu_x_us = pm.Normal(r"$\beta_1$", mu=0, sd=1, shape=len(countries))
        sigma_x_us = pm.HalfCauchy("sigma_x_us", beta=10)
        x_coeff = pm.Normal(r"$\beta_0$", mu_x + x_offset, sigma=sigma_x_coef, shape=len(countries))
        x_ar = pm.Normal(r"$\alpha_1$", mu_ar + ar_offset, sigma=sigma_x_ar, shape=len(countries))

        # Define likelihood
        likelihood = pm.Normal("y",
                               mu=intercept[country_id] + x_coeff[country_id] * x + x_ar[country_id] * x_exergy
                               + mu_x_us[country_id]*x_us,
                               sigma=sigma[country_id],
                               observed=y)
        # Inference!
        trace = pm.sample(draws=500, tune=1000, cores=1, chains=2, target_accept=0.95)

    with model:
        predictions = pm.sample_posterior_predictive(trace, var_names=[r"$\beta_1$", r"$\alpha_0$", r"$\beta_0$",
                                                                       r"$\alpha_1$", "sigma", "y"],
                                                     random_seed=5)

        az.plot_trace(trace, var_names=[r"$\beta_1$", r"$\alpha_0$", r"$\beta_0$", r"$\alpha_1$", "sigma"],
                      divergences=None, legend=True, circ_var_names=countries)
        idata = az.from_pymc3(model=model, posterior_predictive=predictions)

    if register_data:
        for country in countries:
            register(idata, countries.index(country), country)

    return predictions, countries, idata, y