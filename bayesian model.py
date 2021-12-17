import pymc3 as pm
import arviz as az
import numpy as np
from utils.utils import med_regression, get_params, get_pydice_parameters
from utils.utils_bayes import bayesian_model, get_data, generate_scenario, load_predictions, load_posterior
import matplotlib.pyplot as plt
import matplotlib
from DICE.pydice import DICE
import scipy.stats as stats
import os


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


def reg_median(x, y):
    """

    :param x:
    :param y:
    :return: Median regression of x on y, useless.
    """
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


def visu(predictions, countries, idata, y, alpha, param):
    """
    Reproduction of the visualisation of Dice : Measurement without data

    :param predictions: Predictions returned from the function bayesian_model
    :param countries: list of countries for example : ["United States", "France", ...]
    :param idata: element returned by bayesian_model
    :param y: Observed data
    :return: None
    """

    # az.plot_ppc(idata)
    mu_p = predictions["y"]
    lower = [np.sort(mu_p.T[k])[int(len(mu_p.T[k]) * alpha)] for k in range(len(mu_p[0]))]
    upper = [np.sort(mu_p.T[k])[int(len(mu_p.T[k]) * (1 - alpha))] for k in range(len(mu_p[0]))]
    """
    print(mu_p.mean(0))
    print("#################")
    print(mse(mu_p.mean(0), y))
    print(r2(mu_p.mean(0), y))
    print(len(y))
    print("#################")
    """
    fig, axes = plt.subplots(nrows=len(countries), sharex=True, figsize=(20, 20))
    plt.xticks([year for year in range(0, 46, 5)], [1972 + year for year in range(0, 46, 5)])
    for k in range(len(countries)):
        axes[k].grid()
        axes[k].set_ylabel(countries[k])
        if k == 0:
            axes[k].plot(mu_p.mean(0)[k * 46:(k + 1) * 46], label="Predictions of the variations of the TFP", alpha=0.6)
            axes[k].plot(y[k * 46:(k + 1) * 46], ms=4, alpha=0.4, label="Observed variations of the TFP")
        else:
            axes[k].plot(mu_p.mean(0)[k * 46:(k + 1) * 46], alpha=0.6)
            axes[k].plot(y[k * 46:(k + 1) * 46], ms=4, alpha=0.4)
        axes[k].set_ylim(-0.04, 0.04)
        axes[k].fill_between([k for k in range(46)], lower[k * 46:(k + 1) * 46], upper[k * 46:(k + 1) * 46], alpha=0.1,
                             color="blue")

    # ax.vlines([(k) * 46 for k in range(len(countries) + 1)], ymin=min(y), ymax=max(y), color="red")
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="lower center")
    if os.path.exists("sorties/model_{model_name}/fit_plot".format(model_name=param["model_name"])):
        k = len(os.listdir("sorties/model_{model_name}/fit_plot".format(model_name=param["model_name"])))
        fig.savefig("sorties/model_{model_name}/fit_plot/plot_{k}.png".format(model_name=param["model_name"], k=k))
    else:
        if os.path.exists("sorties/model_{model_name}".format(model_name=param["model_name"])):
            os.mkdir("sorties/model_{model_name}/fit_plot".format(model_name=param["model_name"]))
            fig.savefig("sorties/model_{model_name}/fit_plot/fit_plot.png".format(model_name=param["model_name"]))
        else:
            os.mkdir("sorties/model_{model_name}".format(model_name=param["model_name"]))
            os.mkdir("sorties/model_{model_name}/fit_plot".format(model_name=param["model_name"]))
            fig.savefig("sorties/model_{model_name}/fit_plot/fit_plot.png".format(model_name=param["model_name"]))


def get_posterior(delta_a, delta_e, model="model_3", delta_ew = None, plot=False):
    """

    :param model: Model name
    :return:
    """

    all_preds = load_posterior(model)

    countries = os.listdir('sorties/model_1/posterior_data')
    posterior_pres = {country: [] for country in countries}
    returned_post = {country: 0 for country in countries}
    for i, country in enumerate(countries):
        n_draws = len(all_preds[country])
        print("############   {country}   ##################".format(country=country))
        for k in range(n_draws):
            if delta_ew is not None:
                posterior_pres[country] += [
                    all_preds[country][k]["alpha_0"] + all_preds[country][k]["alpha_1"] * delta_a[country] +
                    all_preds[country][k]["beta_0"] * delta_e[country] + all_preds[country][k]["beta_1"] * delta_ew]
            else:
                posterior_pres[country] += [
                    all_preds[country][k]["alpha_0"] + all_preds[country][k]["alpha_1"] * delta_a[country] +
                    all_preds[country][k]["beta_0"] * delta_e[country]]
        if plot:
            print("Median : ", np.sort(posterior_pres[country])[int(0.5 * n_draws)])
            print("0.1 credibility interval : ", np.sort(posterior_pres[country])[int(0.1 * n_draws)])
            print("0.9 credibility interval : ", np.sort(posterior_pres[country])[int(0.9 * n_draws)])
            returned_post[country] = np.sort(posterior_pres[country])[int(0.5 * n_draws)]
    return returned_post


def show_predictions(model="model_3"):
    """

    Shows DICE model with different TFP paths.

    :return:
    """
    all_preds = load_predictions(model)
    countries = os.listdir('sorties/model_1/posterior_data')

    for country in ["United States"]:
        preds = np.array(all_preds[country]).astype(np.float)  # Stored as string
        horizon = len(preds)
        assert [len(preds[t]) == len(preds[0]) for t in
                range(horizon)]  # Verification that all predictions are of same lengths
        nordhaus_params = [get_pydice_parameters("DICE/parameters_nordhaus.json") for _ in range(len(preds))]
        weitzman_params = [get_pydice_parameters("DICE/parameters_weitzman.json") for _ in range(len(preds))]
        n_preds = len(preds)//10
        dices_nord = [DICE(parameters = nordhaus_params[k], tfp=preds[k]) for k in range(n_preds)]
        dices_weitz = [DICE(parameters = weitzman_params[k], tfp=preds[k]) for k in range(n_preds)]

        for i in range(len(preds[0])):
            print("year : ", i)
            for k in range(n_preds):
                dices_nord[k].step()
                dices_weitz[k].step()

        fig = plt.figure()
        plt.title("Economic output with {country} TFP".format(country=country))
        median_nord = [np.sort([dices_nord[k].parameters["output"][t] for k in range(len(dices_nord))])[n_preds//2] for t in range(len(preds[0]))]
        median_weitzman = [np.sort([dices_weitz[k].parameters["output"][t] for k in range(len(dices_nord))])[n_preds // 2]
                       for t in range(len(preds[0]))]

        # plt.plot(dice.parameters["output"], color="red", label="Nordhaus assumptions")
        for k in range(n_preds):
            if k == 0:
                plt.plot(median_nord, color="blue", alpha=1,
                         label="Median simulated output, Nordhaus damage function")
                plt.plot(median_weitzman, color="red", alpha=1,
                         label="Median simulated output, Weitzman damage function")
            else:
                plt.plot(dices_nord[k].parameters["output"], color="blue", alpha=0.01)
                plt.plot(dices_weitz[k].parameters["output"], color="red", alpha=0.01)
        plt.xticks([k for k in range(len(preds[0]))], [str(2011 + k) for k in range(len(preds[0]))], rotation=45)
        plt.ylim(60, 150)
        plt.ylabel("Trillions of $2010")
        plt.legend(loc="best")
        print("Saving figures")
        if os.path.exists("sorties/{model_name}/predictions".format(model_name=model)):
            k = len(os.listdir("sorties/{model_name}/predictions".format(model_name=model)))
            fig.savefig(
                "sorties/{model_name}/predictions/predictions_{country}.png".format(model_name=model, country=country))
        else:
            if os.path.exists("sorties/{model_name}".format(model_name=model)):
                os.mkdir("sorties/{model_name}/predictions".format(model_name=model))
                fig.savefig(
                    "sorties/{model_name}/predictions/predictions_{country}.png".format(model_name=model,
                                                                                        country=country))
            else:
                os.mkdir("sorties/{model_name}".format(model_name=model))
                os.mkdir("sorties/{model_name}/predictions".format(model_name=model))
                fig.savefig(
                    "sorties/{model_name}/fit_plot/predictions_{country}.png".format(model_name=model, country=country))


def plot_emissions(data, param,
                   countries=["United States", "Japan", "Italy", "Germany", "United Kingdom", "France", "Canada"]):
    """

    :param data:
    :param countries:
    :return:
    """
    data = data["x_emissions"]
    data_length = len(data[0])

    assert len(data) == len(countries), "Problem in your arguments."
    assert np.all([len(data[k]) == data_length for k in range(len(data))]), "Not all series are of same length."

    print("### Emissions USA ###")
    print(data[0])
    print("#########")
    fig, axes = plt.subplots(nrows=len(countries), sharex=True, figsize=(20, 20))
    plt.xticks([year for year in range(0, data_length, 5)], [2020 + year for year in range(0, data_length, 5)])

    for k in range(len(countries)):
        axes[k].grid()
        axes[k].set_ylabel(countries[k])
        if k == 0:
            axes[k].plot([data[k][t] / data[k][0] for t in range(data_length)], label="Emissions", alpha=0.6)
        else:
            axes[k].plot([data[k][t] / data[k][0] for t in range(data_length)], alpha=0.6)
        axes[k].set_ylim(0, 1)

    # ax.vlines([(k) * 46 for k in range(len(countries) + 1)], ymin=min(y), ymax=max(y), color="red")
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="lower center")
    if os.path.exists("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"])):
        k = len(os.listdir("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"])))
        fig.savefig(
            "sorties/model_{model_name}/emissions_plot/plot_{k}.png".format(model_name=param["model_name"], k=k))
    else:
        if os.path.exists("sorties/model_{model_name}".format(model_name=param["model_name"])):
            os.mkdir("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"]))
            fig.savefig(
                "sorties/model_{model_name}/emissions_plot/emissions_plot.png".format(model_name=param["model_name"]))
        else:
            os.mkdir("sorties/model_{model_name}".format(model_name=param["model_name"]))
            os.mkdir("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"]))
            fig.savefig(
                "sorties/model_{model_name}/emissions_plot/emissions_plot.png".format(model_name=param["model_name"]))


if __name__ == "__main__":
    print(pm.__version__)
    print(az.__version__)

    delta_a = {"France": 0, "United States": 0, "Japan": 0, "Germany": 0, "United Kingdom": 0, "Italy": 0, "Canada": 0}
    delta_e = {"France": -0.143, "United States": -0.109, "Japan": -0.065, "Germany": -0.097, "United Kingdom": -0.112,
               "Italy": -0.107, "Canada": -0.096}
    delta_a = get_posterior(delta_a=delta_a, delta_e=delta_e, model="model_7", delta_ew=-0.1, plot=False)
    # delta_e hyp 2021:
    delta_e = {"France": 0.07, "United States": 0.07, "Japan": 0.07, "Germany": 0.07, "United Kingdom": 0.07,
               "Italy": 0.07, "Canada": 0.07}
    get_posterior(delta_a=delta_a, delta_e=delta_e, model="model_3", plot=True)
    print("Computing DICE")
    show_predictions(model="model_3")
    print("Computing DICE with world exergy")
    show_predictions(model="model_4")
    params = get_params()
    horizon = 29
    alpha = 0.1
    for param in params:
        if param["compute"]:
            assert np.all(
                np.array([key in list(param.keys()) for key in ["mu_beta_1", "sigma_mu_beta_1", "sigma_beta_1"]]) ==
                param["include beta 1"])
            # Tune : Number of steps to reach
            tune = param["tuning steps"]
            # Number of draws from the posterior density
            draws = param["draws"]
            # Which model should be used (eg described in : 2.3.3 or 2.3.4)
            # Parameters
            include_world_exergy = bool(param["include beta 1"])
            # Scenario to be used
            scenario = "scenario_1.json"
            # Countries to be considered
            countries = ["United States", "Japan", "Italy", "Germany", "United Kingdom", "France", "Canada"]

            data_pred = generate_scenario(scenario, countries, horizon)
            data = get_data()
            plot_emissions(data_pred, param)

            predictions, countries, idata, y, trace, fig = bayesian_model(data=data, data_pred=data_pred,
                                                                          register_data=True,
                                                                          tune=tune, draws=draws,
                                                                          include_world_exergy=include_world_exergy,
                                                                          hyperparams=param)

            if os.path.exists("sorties/model_{model_name}/posterior".format(model_name=param["model_name"])):
                k = len(os.listdir("sorties/model_{model_name}/posterior".format(model_name=param["model_name"])))
                fig.savefig(
                    "sorties/model_{model_name}/posterior/posterior_{k}".format(model_name=param["model_name"], k=k))
            else:
                os.mkdir("sorties/model_{model_name}/posterior".format(model_name=param["model_name"]))
                k = 0
                fig.savefig(
                    "sorties/model_{model_name}/posterior/posterior_{k}".format(model_name=param["model_name"], k=k))
            fig, axs = plt.subplots(3, 3, figsize=(14, 14))
            for k in range(3):
                for i in range(3):
                    axs[k, i].set_axis_off()
            coords = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [0, 2], [1, 2], [2, 1]]
            visu(predictions, countries, idata, data["y"], alpha=alpha, param=param)
            plt.figure()
            for country in countries:
                axs[coords[countries.index(country)][0], coords[countries.index(country)][1]].grid()
                axs[coords[countries.index(country)][0], coords[countries.index(country)][1]].set_ylim(-0.038, 0.038)
                axs[coords[countries.index(country)][0], coords[countries.index(country)][1]].set_axis_on()
                axs[coords[countries.index(country)][0], coords[countries.index(country)][1]].set_title(country)
                lower = [np.sort([trace["Y_{t}".format(t=t)][k][countries.index(country)] for k in range(draws)])[
                             int(alpha * draws)] for t in
                         range(horizon)]
                upper = [np.sort([trace["Y_{t}".format(t=t)][k][countries.index(country)] for k in range(draws)])
                         [int((1 - alpha) * draws)] for t in
                         range(horizon)]

                axs[coords[countries.index(country)][0], coords[countries.index(country)][1]].fill_between(
                    [k for k in range(horizon)], lower, upper, color="blue", alpha=0.1)

                axs[coords[countries.index(country)][0], coords[countries.index(country)][1]].plot(
                    [np.mean([trace["Y_{t}".format(t=t)][k][countries.index(country)] for k in range(draws)]) for t in
                     range(horizon)], color="red")
            fig.tight_layout()
            if os.path.exists("sorties/model_{model_name}".format(model_name=param["model_name"])):
                if os.path.exists("sorties/model_{model_name}/predictions".format(model_name=param["model_name"])):
                    k = len(os.listdir("sorties/model_{model_name}/predictions".format(model_name=param["model_name"])))
                    fig.savefig(
                        "sorties/model_{model_name}/predictions/countries_{k}.png".format(
                            model_name=param["model_name"],
                            k=k))
                else:
                    os.mkdir("sorties/model_{model_name}/predictions".format(model_name=param["model_name"]))
                    fig.savefig(
                        "sorties/model_{model_name}/predictions/countries_{k}.png".format(
                            model_name=param["model_name"],
                            k=0))
            else:
                os.mkdir("sorties/model_{model_name}".format(model_name=param["model_name"]))
        plt.close("all")
