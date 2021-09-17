import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from utils.utils import get_tfp, get_data, get_exergy_iea, compute_tfp_level, get_energy_iea, get_exergy, \
    get_exergy_coefs, gen_exergy, diff
from scenario import scenario
import json
import scipy
from statsmodels.tsa.stattools import adfuller


def var(data):
    return ([100 * (data[t] - data[t - 1]) / data[t - 1] for t in range(1, len(data))])



if __name__ == "__main__":
    to_plot = True

    horizon = 100
    data_scenario = scenario.gen_scenario(horizon)

    with open("data/scenario.json", "w") as json_file:
        json.dump(data_scenario, json_file)
    
    Exergy_US = get_exergy_iea()
    c_e = Exergy_US[0]
    Exergy_US = [e / c_e for e in Exergy_US]
    Exergy_US = var(Exergy_US)
    # Exergy_US = var(Exergy_US)
    tfp = get_tfp("data/tfp")[2:]

    tfp_level = compute_tfp_level(tfp)
    tfp_level = [t * 1.782614 for t in tfp_level]  # 1.782614 correspond à la valeur de référence de la TFP
    exergy_level = compute_tfp_level(Exergy_US)

    import pandas as pd

    mod = sm.OLS(tfp_level, exergy_level)
    reg = mod.fit()

    print("tfp", adfuller(tfp_level))
    print("diff tfp", adfuller(diff(tfp)))
    print('Exergy', adfuller(exergy_level))
    print("diff Exergy", adfuller(Exergy_US))
    print(adfuller(reg.resid))

    n_obs = 0

    df = pd.DataFrame(np.array([diff(tfp_level)[:-1], diff(exergy_level)[1:]]).T, columns=["TFP", "exergy"])
    model = VAR(df)

    res = model.fit(1)
    y = diff(tfp_level)[:-1]
    var_y = np.sum([(y[t] - np.mean(y)) ** 2 for t in range(len(y))])
    print(res.resid["TFP"].values ** 2)
    print(res.resid["TFP"].values ** 2)
    # PIB_US = var(PIB_US)
    print(res.summary())
    print("R2 :", 1 - np.sum(res.resid["TFP"].values ** 2) / np.sum(var_y))
    print(sm.stats.acorr_ljungbox(res.resid["TFP"], lags=[1, 2, 3, 4, 5, 10], return_df=True))
    print(sm.stats.stattools.jarque_bera(res.resid["TFP"]))

    """
    plt.figure()
    plt.plot(res.fittedvalues["TFP"], label="Predicted total factor productivity")
    plt.plot(diff(tfp_level)[:-1], label="Observed total factor productivity")
    plt.legend()
    plt.draw()
    
    tfp_pred = [diff(tfp_level)[-n_obs+1]]
    diff_exergy = diff(exergy_level[-n_obs-1:])
    print(len(diff_exergy))
    for exergy in diff_exergy:
        tfp_pred += [res.coefs_exog[0][0] + res.coefs[0][0][0] * tfp_pred[-1] + res.coefs[0][0][1]*float(exergy)]
    print("RMSE : ", np.mean([(tfp_pred[k] - diff(tfp_level)[-n_obs - 1 + k])**2 for k in range(len(tfp_pred))]))
    print("RMSE : ", np.mean([np.abs(tfp_pred[k] - diff(tfp_level)[-n_obs - 1 + k])for k in range(len(tfp_pred))]))

    plt.figure()
    plt.plot(tfp_pred, label="Out of sample prediction")
    plt.plot(diff(tfp_level)[-n_obs-1:], label="truc")
    plt.legend()
    plt.show()
    
    # Source : https://fred.stlouisfed.org/series/RTFPNAUSA632NRUG
    # Source : https://www.frbsf.org/economic-research/publications/working-papers/2012/19/

    X = [[1, Exergy_US[i]] for i in range(len(Exergy_US))]
    Y = tfp
    mod = sm.OLS(Y, X)
    reg = mod.fit()
    if to_plot:
        print(reg.summary().as_latex())
        plt.figure()
        plt.plot(Y, label="TFP variations", color="red")
        plt.plot(reg.fittedvalues, label="Model predictions", marker="o", color="blue")
        plt.xticks([k for k in range(0, len(reg.fittedvalues), 5)], [str(1950 + k) for k in range(0, len(reg.fittedvalues), 5)])
        plt.legend()
        plt.draw()
        plt.figure()
        measurements = np.random.normal(loc=20, scale=5, size=100)
        scipy.stats.probplot(measurements, dist="norm", plot=plt)
        plt.show()

    plt.figure()
    plt.plot(reg.resid)
    plt.draw()
    
    """
    energy_us = get_energy_iea()
    scenario_paris = scenario.gen_scenario(100, scenari="Paris")
    scenario_bau = scenario.gen_scenario(100, scenari="bau")
    scenario_peak = scenario.gen_scenario(100, scenari="bau", peak=True)
    exergy_coef = get_exergy_coefs()
    exergy_scenario_paris = gen_exergy(exergy_coef, energy_us[0]["2020"], scenario_paris, horizon + 1, c_e)
    exergy_scenario_bau = gen_exergy(exergy_coef, energy_us[0]["2020"], scenario_bau, horizon + 1, c_e)
    exergy_scenario_peak = gen_exergy(exergy_coef, energy_us[0]["2020"], scenario_peak, horizon + 1, c_e)
    nb_scenarios = 200
    tfp_pred_paris = [[tfp_level[-2], tfp_level[-1]] for _ in range(nb_scenarios)]
    tfp_pred_bau = [[tfp_level[-2], tfp_level[-1]] for _ in range(nb_scenarios)]
    tfp_pred_peak = [[tfp_level[-2], tfp_level[-1]] for _ in range(nb_scenarios)]
    resid_var = np.var(reg.resid)
    for i in range(nb_scenarios):
        for t in range(horizon):
            tfp_pred_paris[i] += [tfp_pred_paris[i][-1] + res.coefs_exog[0][0] + res.coefs[0][0][0]*(tfp_pred_paris[i][-1] - tfp_pred_paris[i][-2]) +res.coefs[0][0][1]*exergy_scenario_paris[t] + np.random.normal(0, resid_var)]
            tfp_pred_bau[i] += [tfp_pred_bau[i][-1] + res.coefs_exog[0][0] + res.coefs[0][0][0] * (
                        tfp_pred_bau[i][-1] - tfp_pred_bau[i][-2]) + res.coefs[0][0][1] * exergy_scenario_bau[
                                      t] + np.random.normal(0, resid_var)]
            tfp_pred_peak[i] += [tfp_pred_peak[i][-1] + res.coefs_exog[0][0] + res.coefs[0][0][0] * (
                    tfp_pred_peak[i][-1] - tfp_pred_peak[i][-2]) + res.coefs[0][0][1] * exergy_scenario_peak[
                                    t] + np.random.normal(0, resid_var)]
    with open("data/tfp_nordhaus.csv", "r") as csv_file:
        raw_tfp_values = csv.reader(csv_file, delimiter=";")
        tfp_nordhaus = []
        for row in raw_tfp_values:
            tfp_nordhaus += [float(row[0].replace(",", "."))]
    with open("data/tfp_ACPR.csv", "r") as csv_file:
        raw_tfp_values = csv.reader(csv_file, delimiter=";")
        raw_tfp_acpr = []
        for row in raw_tfp_values:
            raw_tfp_acpr += [float(row[0].replace(",", "."))]
    tfp_pred_paris = np.array(tfp_pred_paris).T
    tfp_pred_bau = np.array(tfp_pred_bau).T
    tfp_pred_peak = np.array(tfp_pred_peak).T

    alpha=0.01
    plt.figure()
    plt.plot([np.sort([tfp for tfp in tfp_pred_bau[t]])[int(nb_scenarios/2)] for t in range(horizon)], color="blue")
    plt.plot([np.sort([tfp for tfp in tfp_pred_bau[t]])[int(nb_scenarios*alpha)] for t in range(horizon)], color="blue", label="Buisiness as usual scenario")
    plt.plot([np.sort([tfp for tfp in tfp_pred_bau[t]])[int(nb_scenarios * (1-alpha))] for t in range(horizon)],
             color="blue")


    plt.plot([np.sort([tfp for tfp in tfp_pred_paris[t]])[int(nb_scenarios/2)] for t in range(horizon)], color="red")
    plt.plot([np.sort([tfp for tfp in tfp_pred_paris[t]])[int(nb_scenarios*alpha)] for t in range(horizon)], color="red", label="Under Paris agreement scenario")
    plt.plot([np.sort([tfp for tfp in tfp_pred_paris[t]])[int(nb_scenarios * (1-alpha))] for t in range(horizon)],
             color="red")

    plt.scatter([k*5 for k in range(len(tfp_nordhaus))], tfp_nordhaus, color="blue", label="Nordhaus assumptions")
    plt.scatter([k*5 for k in range(len(raw_tfp_acpr))], raw_tfp_acpr, color="red", label="ACPR assumptions")
    plt.legend()

    with open("tfp_pred_paris.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow([str(round(np.sort([tfp for tfp in tfp_pred_paris[t]])[int(nb_scenarios/2)],3)).replace(".", ",") for t in range(0, horizon, 5)])

    plt.show()