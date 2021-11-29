import pymc3 as pm
import arviz as az
import numpy as np
from utils.utils import med_regression, get_params
from utils.utils_bayes import bayesian_model, get_data, predict, generate_scenario, load_predictions
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

    #az.plot_ppc(idata)
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


def show_predictions(model="model_3", alpha=0.05):
    """

    Shows DICE model with different TFP paths.

    :return:
    """
    all_preds = load_predictions(model)
    countries = os.listdir('sorties/model_1/posterior_data')


    for country in countries:
            preds = np.array(all_preds[country]).astype(np.float)  #Stored as string
            horizon = len(preds)
            assert [len(preds[t]) == len(preds[0]) for t in range(horizon)] #Verification that all predictions are of same lengths


            dices = [DICE(tfp=preds[k]) for k in range(len(preds))]
            dice = DICE()
            for _ in range(len(preds[0])):
                for k in range(len(preds)):
                    dices[k].step()
                #dice.step()


            fig = plt.figure()
            plt.title("Economic output with {country} TFP".format(country=country))
            #plt.plot(dice.parameters["output"], color="red", label="Nordhaus assumptions")
            for k in range(len(preds)):
                if k == 0:
                    plt.plot(dices[k].parameters["output"], color="blue", alpha=0.01, label="Simulated output, Nordhaus damage function")
                else:
                    plt.plot(dices[k].parameters["output"], color="blue", alpha=0.01)
            plt.xticks([k for k in range(len(preds[0]))], [str(2021 + k) for k in range(len(preds[0]))], rotation=45)
            plt.xlabel("Years of prediction")
            plt.ylabel("Trillions of $2010")
            plt.legend(loc="best"                                                                                                                                                                                                                       )
            plt.draw()
            if os.path.exists("sorties/{model_name}/predictions".format(model_name=model)):
                k = len(os.listdir("sorties/{model_name}/predictions".format(model_name=model)))
                fig.savefig(
                    "sorties/{model_name}/predictions/predictions_{country}.png".format(model_name=model, country=country))
            else:
                if os.path.exists("sorties/{model_name}".format(model_name=model)):
                    os.mkdir("sorties/{model_name}/predictions".format(model_name=model))
                    fig.savefig(
                        "sorties/{model_name}/predictions/predictions_{country}.png".format(model_name=model, country=country))
                else:
                    os.mkdir("sorties/{model_name}".format(model_name=model))
                    os.mkdir("sorties/{model_name}/predictions".format(model_name=model))
                    fig.savefig(
                        "sorties/{model_name}/fit_plot/predictions_{country}.png".format(model_name=model, country=country))

def plot_emissions(data, param, countries = ["United States", "Japan", "Italy", "Germany", "United Kingdom", "France", "Canada"]):
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
            axes[k].plot([data[k][t]/data[k][0] for t in range(data_length)], label="Emissions", alpha=0.6)
        else:
            axes[k].plot([data[k][t]/data[k][0] for t in range(data_length)], alpha=0.6)
        axes[k].set_ylim(0, 1)

    # ax.vlines([(k) * 46 for k in range(len(countries) + 1)], ymin=min(y), ymax=max(y), color="red")
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="lower center")
    if os.path.exists("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"])):
        k = len(os.listdir("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"])))
        fig.savefig("sorties/model_{model_name}/emissions_plot/plot_{k}.png".format(model_name=param["model_name"], k=k))
    else:
        if os.path.exists("sorties/model_{model_name}".format(model_name=param["model_name"])):
            os.mkdir("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"]))
            fig.savefig("sorties/model_{model_name}/emissions_plot/emissions_plot.png".format(model_name=param["model_name"]))
        else:
            os.mkdir("sorties/model_{model_name}".format(model_name=param["model_name"]))
            os.mkdir("sorties/model_{model_name}/emissions_plot".format(model_name=param["model_name"]))
            fig.savefig("sorties/model_{model_name}/emissions_plot/emissions_plot.png".format(model_name=param["model_name"]))

if __name__ == "__main__":
    print(pm.__version__)
    print(az.__version__)

    show_predictions(model="model_3")
    show_predictions(model="model_4")
    params = get_params()
    horizon = 29
    alpha = 0.1
    for param in params:
        if param["compute"]:
            assert np.all(np.array([key in list(param.keys()) for key in ["mu_beta_1", "sigma_mu_beta_1", "sigma_beta_1"]]) == param["include beta 1"])
            # Tune : Number of steps to reach
            tune = param["tuning steps"]
            # Number of draws from the posterior density
            draws = param["draws"]
            # Which model should be used (eg described in : 2.3.3 or 2.3.4)
            # Parameters
            include_world_exergy = bool(param["include beta 1"])
            # Scenario to be used
            scenario = "scenario_2.json"
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
                axs[coords[countries.index(country)][0], coords[countries.index(country)][1]].set_xticks([year for year in range(0, 30, 5)], [2020 + year for year in range(0, 30, 5)])
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
                        "sorties/model_{model_name}/predictions/countries_{k}.png".format(model_name=param["model_name"],
                                                                                          k=k))
                else:
                    os.mkdir("sorties/model_{model_name}/predictions".format(model_name=param["model_name"]))
                    fig.savefig(
                        "sorties/model_{model_name}/predictions/countries_{k}.png".format(model_name=param["model_name"],
                                                                                          k=0))
            else:
                os.mkdir("sorties/model_{model_name}".format(model_name=param["model_name"]))
        plt.close("all")