import pymc3 as pm
import numpy as np
import arviz as az
from theano import tensor as tt
from utils.utils import get_countries, diff, countries_energy, tfp, get_scenario
import csv
import os

def generate_scenario(scenario, countries, horizon):

    x_exergy_var = []
    x_emissions = []
    for country in countries:
        x_exergy_c, x_emissions_c = get_scenario(scenario=scenario, country=country)
        x_exergy_var += [x_exergy_c]
        x_emissions += [x_emissions_c]

    x_us_var = [np.sum([x_exergy_var[c][t] for c in range(len(countries))]) for t in range(len(x_exergy_var[0]))]
    x_exergy_var = np.array([diff(np.log(x_exergy_var[c])) for c in range(len(x_exergy_var))]).T
    x_us_var = diff([np.log(x) for x in x_us_var])
    data_pred = {"x_pred": x_exergy_var, "x_us_pred": x_us_var, "x_emissions": x_emissions}

    assert horizon < len(x_exergy_var)
    assert horizon < len(x_us_var)

    return data_pred

def load_predictions(model, parent_dir = ""):
    countries = os.listdir(parent_dir + "sorties/{model}/posterior_data".format(model=model))
    preds = {country : [] for country in countries}
    for country in countries:
        list_estimations = os.listdir(parent_dir + "sorties/{model}/posterior_data/{country}".format(model=model, country=country))
        with open(parent_dir + "sorties/{model}/posterior_data/{country}/".format(model=model, country=country) + list_estimations[-1], "r") as csv_file:
            rows = csv.reader(csv_file)
            for i, row in enumerate(rows):
                if i == 0:
                    var_names = row
                    list_indices = ["Y_" in var for var in var_names]
                else:
                    preds[country] += [[row[k] for k,b in enumerate(list_indices) if b]]
    return preds



def predict(idata, traj_e, delta_a=-0.02, horizon=30, sample_size=1000, include_world_exergye=False):
    predictions = []
    return predictions

def pred_posterior(idata, country_id, delta_a, delta_e, delta_e_w, world_exergy=True):
    alpha_0 = [idata["posterior_predictive"][r"$\alpha_0$"][0][t][country_id] for t in
               range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    alpha_1 = [idata["posterior_predictive"][r"$\alpha_1$"][0][t][country_id] for t in
               range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    beta_0 = [idata["posterior_predictive"][r"$\beta_0$"][0][t][country_id] for t in
              range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    if world_exergy:
        beta_1 = [idata["posterior_predictive"][r"$\beta_1$"][0][t][country_id] for t in
                  range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    sigma = [idata["posterior_predictive"]["sigma"][0][t][country_id] for t in
             range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]

    if world_exergy:
        post = np.array(alpha_0) + np.array([alpha_1[k]*delta_a for k in range(len(alpha_1))]) + np.array([beta_0[k]*delta_e for k in range(len(alpha_1))]) + np.array([beta_1[k]*delta_e_w for k in range(len(alpha_1))]) + np.random.normal(0, sigma)
    else:
        post = np.array(alpha_0) + np.array([alpha_1[k] * delta_a for k in range(len(alpha_1))]) + np.array(
            [beta_0[k] * delta_e for k in range(len(alpha_1))]) + np.random.normal(0, sigma)
    #equivalent of np.add : np.array([1,2,3]) + np.array([1,2,3]) = array([2,4,6])
    return post

def register(idata, country_id, country_name, tuning, draw, param, world_exergy=True, horizon = 29):
    """
    Register data. Should not be used alone.
    :param idata:
    :param country_id:
    :param country_name:
    :param world_exergy:
    :return: None
    """
    alpha_0 = [idata["posterior_predictive"][r"$\alpha_0$"][0][t][country_id] for t in
               range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    alpha_1 = [idata["posterior_predictive"][r"$\alpha_1$"][0][t][country_id] for t in
               range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    beta_0 = [idata["posterior_predictive"][r"$\beta_0$"][0][t][country_id] for t in
              range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    if world_exergy:
        beta_1 = [idata["posterior_predictive"][r"$\beta_1$"][0][t][country_id] for t in
                  range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]
    sigma = [idata["posterior_predictive"][r"$\sigma$"][0][t][country_id] for t in
             range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]

    posterior_Y = []
    for h in range(horizon):
        posterior_Y += [[idata["posterior_predictive"]["Y_{h}".format(h=h)][0][t][country_id] for t in
              range(len(idata["posterior_predictive"][r"$\alpha_0$"][0]))]]
    if world_exergy:
        names = ["alpha_0", "alpha_1", "beta_0", "beta_1", r"$\sigma$"] + ["Y_{h}".format(h=h) for h in range(horizon)]
        rows = np.array([alpha_0, alpha_1, beta_0, beta_1, sigma] + [posterior_Y[h] for h in range(horizon)]).T
    else:
        names = ["alpha_0", "alpha_1", "beta_0", r"$\sigma$"] + ["Y_{h}".format(h=h) for h in range(horizon)]
        rows = np.array([alpha_0, alpha_1, beta_0, sigma] + [posterior_Y[h] for h in range(horizon)]).T


    if os.path.exists("sorties/model_{model_number}/posterior_data/{country}".format(model_number=param["model_name"], country=country_name)):
        k = len(os.listdir("sorties/model_{model_number}/posterior_data/{country}".format(model_number=param["model_name"], country=country_name)))
        with open("sorties/model_{model_number}/posterior_data/{country}/posterior_predictive_{world_exergy}_tune_{tune}_draws_{draws}_model_{k}.csv".format(model_number=param["model_name"], tune=tuning, draws=draw, k=k, country=country_name, world_exergy=world_exergy), "w", newline="") as csv_file:
            write = csv.writer(csv_file)
            write.writerow(names)
            for row in rows:
                write.writerow(row)
    else:
        if os.path.exists("sorties/model_{model_number}/posterior_data/".format(model_number=param["model_name"])):
            os.mkdir(
                "sorties/model_{model_number}/posterior_data/{country}".format(model_number=param["model_name"], country=country_name))
        else:
            if os.path.exists("sorties/model_{model_number}".format(model_number=param["model_name"])):
                os.mkdir(
                    "sorties/model_{model_number}/posterior_data".format(model_number=param["model_name"]))
                os.mkdir(
                    "sorties/model_{model_number}/posterior_data/{country}".format(model_number=param["model_name"],
                                                                                  country=country_name))
            else:
                os.mkdir("sorties/model_{model_number}".format(model_number=param["model_name"]))
                os.mkdir(
                    "sorties/model_{model_number}/posterior_data".format(model_number=param["model_name"]))
                os.mkdir(
                    "sorties/model_{model_number}/posterior_data/{country}".format(model_number=param["model_name"], country=country_name))

        with open("sorties/model_{model_number}/posterior_data/{country}/posterior_predictive_{world_exergy}_tune_{tune}_draws_{draws}_model_{k}.csv".format(model_number=param["model_name"], tune=tuning, draws=draw, k=0, country=country_name, world_exergy=world_exergy), "w", newline="") as csv_file:
            write = csv.writer(csv_file)
            write.writerow(names)
            for row in rows:
                write.writerow(row)

def get_data(countries = None):
    """

    :param countries: List of countries to get data from
    :return: a dictionary with all the necessary data, this dictionnary as the following keys :
    "y": Observed data
    "x_us": variation of world exergy. Used only if required in the model.
    "x_exergy" : variation of exergy for each country.
    "x" : Variation of tfp with a lag.
    "country_id": id of the countries, required for the hierarchical bayesian model.
    "countries": List of countries.
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

def bayesian_model(data, data_pred, register_data = True, tune=500, draws=1000, include_world_exergy = False, horizon=30, hyperparams=1):
    """

    :param data: Data dictionnary as returned by the function get_data()
    :param data_pred: Prediction data, should have the following keys : 'x_pred', 'x_us_pred'.
    :param register_data: Boolean, should the posterior densities be registered ?
    :param tune: Number of tuning step before starting to draw. Markov chains should have convergenced before the last tuning step.
    :param draws: Number of draws from the posterior density.
    :param include_world_exergy: Boolean, which model should be used (eg : described in section 2.3.3 or 2.3.4 ?)
    :param horizon: horizon of prediction, should be lower than the length of x_pred and x_us_pred

    :return: predictions, countries, idata, y, trace
    which are :
        predictions : The fitted values
        countries : List of the countries
        idata :
        y : Observed values.
        trace : Trace as return by pymc3.
    """

    countries, country_id, x, x_exergy, x_us, y = data["countries"], data["country_id"], data["x"], data["x_exergy"], data["x_us"], data["y"]
    x_pred, x_us_pred = data_pred["x_pred"], data_pred["x_us_pred"]

    with pm.Model() as model:
        # Define priors
        sigma = pm.HalfCauchy(r"$\sigma$", beta=hyperparams["sigma"], shape=len(countries))
        intercept = pm.Normal(r"$\alpha_0$", hyperparams["mu_alpha_0"], sigma=hyperparams["sigma_mu_alpha_0"], shape=len(countries))

        #x_offset = pm.Normal('x_offset', mu=0, sigma=0.1, shape=len(countries))
        mu_x = pm.Normal("prior exergy coefficient", hyperparams["mu_beta_0"], sigma=hyperparams["sigma_mu_beta_0"])
        #mu_x = pm.Uniform("prior exergy coefficient", lower=-1, upper=1)
        sigma_x_coef = pm.HalfCauchy("sigma_exergy_coef", beta=hyperparams["sigma_beta_0"])

        #ar_offset = pm.Normal('ar_offset', mu=0, sigma=0.1, shape=len(countries))
        mu_ar = pm.Normal(r"$\mu_{\alpha_1}$", hyperparams["mu_alpha_1"], sigma=hyperparams["sigma_mu_alpha_1"])
        sigma_x_ar = pm.HalfCauchy("sigma_ar_coef", beta=hyperparams["sigma_alpha_1"])

        if include_world_exergy:
            sigma_x_us = pm.HalfCauchy("sigma_x_us", beta=hyperparams["sigma_beta_1"])
            mu_x_us = pm.Normal("mu_x_us", mu=hyperparams["mu_beta_1"], sd=hyperparams["sigma_mu_beta_1"], shape=len(countries))
            x_us_coef = pm.Normal(r"$\beta_1$", mu=mu_x_us, sigma=sigma_x_us, shape=len(countries))


        #x_coeff = pm.Normal(r"$\beta_0$", mu_x + x_offset, sigma=sigma_x_coef, shape=len(countries))
        #x_ar = pm.Normal(r"$\alpha_1$", mu_ar + ar_offset, sigma=sigma_x_ar, shape=len(countries))
        x_coeff = pm.Normal(r"$\beta_0$", mu_x, sigma=sigma_x_coef, shape=len(countries))
        x_ar = pm.Normal(r"$\alpha_1$", mu_ar, sigma=sigma_x_ar, shape=len(countries))
        # Define likelihood
        #print(x_coeff * x)
        if include_world_exergy:
            if hyperparams["posterior distribution"] == "Cauchy":
                likelihood = pm.Cauchy("y",
                                   alpha=intercept[country_id] + x_coeff[country_id] * x + x_ar[country_id] * x_exergy
                                      + x_us_coef[country_id] * x_us,
                                   beta=sigma[country_id],
                                   observed=y)
            else:
                likelihood = pm.Normal("y",
                                   mu=intercept[country_id] + x_coeff[country_id] * x + x_ar[country_id] * x_exergy
                                      + x_us_coef[country_id] * x_us,
                                   sigma=sigma[country_id],
                                   observed=y)
            for t in range(horizon):
                y_preds = [x_exergy[(k+1)*len(x_exergy)//7 - 1] for k in range(len(countries))]
                if hyperparams["posterior distribution"] == "Cauchy":
                    y_preds += [pm.Cauchy("Y_{t}".format(t=t),
                                          alpha=intercept + x_coeff * x_pred[t] + x_ar * y_preds[-1]
                                             + x_us_coef * x_us_pred[t],
                                          beta=sigma, shape=(len(countries),))]
                else:
                    y_preds += [pm.Normal("Y_{t}".format(t=t),
                                      mu=intercept + x_coeff * x_pred[t] + x_ar * y_preds[-1]
                                         + x_us_coef * x_us_pred[t],
                                      sigma=sigma, shape=(len(countries),))]
        else :
            if hyperparams["posterior distribution"] == "Cauchy":
                likelihood = pm.Cauchy("y",
                                       alpha=intercept[country_id] + x_coeff[country_id] * x + x_ar[country_id] * x_exergy,
                                       beta=sigma[country_id],
                                       observed=y)
            else:
                likelihood = pm.Normal("y",
                                   mu=intercept[country_id] + x_coeff[country_id] * x + x_ar[country_id] * x_exergy,
                                   sigma=sigma[country_id],
                                   observed=y)
            for t in range(horizon):
                y_preds = [x_exergy[(k + 1) * len(x_exergy) // 7 - 1] for k in range(len(countries))]
                if hyperparams["posterior distribution"] == "Cauchy":
                    y_preds += [pm.Cauchy("Y_{t}".format(t=t),
                                          alpha=intercept + x_coeff * x_pred[t] + x_ar * y_preds[-1],
                                          beta=sigma, shape=(len(countries),))]
                else:
                    y_preds += [pm.Normal("Y_{t}".format(t=t),
                                          mu=intercept + x_coeff * x_pred[t] + x_ar * y_preds[-1],
                                          sigma=sigma, shape=(len(countries),))]
        # Inference!
        trace = pm.sample(draws=draws, tune=tune, cores=1, chains=2, target_accept=0.95, progressbar=True)

    with model:
        if include_world_exergy:
            predictions = pm.sample_posterior_predictive(trace, var_names=[r"$\beta_1$", r"$\alpha_0$", r"$\beta_0$",
                                                                       r"$\alpha_1$", r"$\sigma$", "y"] + ["Y_{t}".format(t=t) for t in range(horizon)],
                                                     random_seed=5)
            axes = az.plot_trace(trace, var_names=[r"$\beta_1$", r"$\alpha_0$", r"$\beta_0$", r"$\alpha_1$", r"$\sigma$"],
                          divergences=None, legend=False, circ_var_names=countries)
            fig = axes.ravel()[0].figure
        else:
            predictions = pm.sample_posterior_predictive(trace,
                                                             var_names=[r"$\alpha_0$", r"$\beta_0$",
                                                                        r"$\alpha_1$", r"$\sigma$", "y"] + ["Y_{t}".format(t=t) for t in range(horizon)],
                                                             random_seed=5)
            axes = az.plot_trace(trace, var_names=[r"$\alpha_0$", r"$\beta_0$", r"$\alpha_1$", r"$\sigma$"],
                          divergences=None, legend=False, circ_var_names=countries)
            fig = axes.ravel()[0].figure

        idata = az.from_pymc3(model=model, posterior_predictive=predictions)

    if register_data:
        for country in countries:
            register(idata, countries.index(country), country, world_exergy=include_world_exergy, tuning=tune, draw=draws, param=hyperparams)

    return predictions, countries, idata, y, trace, fig

