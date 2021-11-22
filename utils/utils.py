import csv
import numpy as np
import json
import os
from scipy.stats import ks_2samp, norm

def med_regression(x,y, n_deciles=3, alpha=0.5):
    """

    :param x:
    :param y:
    :return:
    """

    x = list(x)
    y=list(y)

    x_sorted =[np.sort(x)[int(k*len(x)/n_deciles) : int((k+1)*len(x)/n_deciles)] for k in range(n_deciles)]
    y_sorted = [[y[x.index(x_value)] for x_value in x_sorted[k]] for k in range(n_deciles)]

    med_x = [np.sort(x)[int(len(x)*0.5)] for x in x_sorted]
    med_y = [np.sort(y)[int(len(y)*alpha)] for y in y_sorted]

    return((med_y[-1] - med_y[0])/(med_x[-1] - med_x[0]), (np.sum([med_y[k] for k in range(n_deciles)]) - np.sum([med_x[k] for k in range(n_deciles)]) ) / n_deciles)

def get_countries(path="data/iea/tfp_countries.csv"):
    countries = []
    with open(path, 'r') as csv_file:
        rows = csv.reader(csv_file)
        for i, row in enumerate(rows):
            if i == 0:
                countries = row
    return countries

def get_params(params_path="data/hyperparams.json"):
    with open(params_path, "r") as json_file:
        hyperparams = json.load(json_file)
    return hyperparams

def countries_energy(exergy_coefs_path = "data/iea/exergy_coefs.json", energy_path="data/iea/IEA_all_countries/countries_energy_balance.csv", country="Belgium", starting_year = 1971, end_year = 2020):
    years = [starting_year + k for k in range(end_year - starting_year)]
    exergy_coef = get_exergy_coefs(exergy_coefs_path)
    energy_data = {"country": {country: {year : {} for year in years}}}
    with open(energy_path, "r") as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            if ("Total energy supply (ktoe)" in row[5] or (
                    "Electricity output (GWh)" in row[5] and "Renewable sources" in row[4])) and country == row[0]:
                if ("Electricity output (GWh)" in row[5] and "Renewable sources" in row[4]):
                    for year in range(len(years)):
                        energy_data["country"][country][starting_year + year].update(
                            {
                                    "Renewable sources": float(row[6 - 1971 + starting_year + year]) * 85.9845 / 1000}
                        )
                else:
                    if "Coal, peat and oil shale" in row[4]:
                        for year in range(len(years)):
                            energy_data["country"][country][starting_year + year].update(
                                {
                                    "Coal": float(row[6 - 1971 + starting_year + year])}
                            )
                    if "Natural gas" in row[4]:
                        for year in range(len(years)):
                            energy_data["country"][country][starting_year + year].update(
                                {
                                    "Natural gas": float(row[6 - 1971 + starting_year + year])}
                            )

                    if "Crude, NGL and feedstocks" in row[4]:
                        for year in range(len(years)):
                            energy_data["country"][country][starting_year + year].update(
                                {
                                    "Crude oil": float(row[6 - 1971 + starting_year + year])}
                            )
                    if "Nuclear" in row[4]:
                        for year in range(len(years)):
                            energy_data["country"][country][starting_year + year].update(
                                {
                                    "Nuclear": float(row[6 - 1971 + starting_year + year])}
                            )
                    if "Renewables and waste" in row[4]:
                        for year in range(len(years)):
                            energy_data["country"][country][starting_year + year].update(
                                {
                                    "Renewables and waste": float(row[6 - 1971 + starting_year+ year])}
                            )
                    if "Oil products" in row[4]:
                        for year in range(len(years)):
                            energy_data["country"][country][starting_year + year].update(
                                {
                                    "Oil products": float(row[6 - 1971 + starting_year + year])}
                            )

        for year in years:
            energy_data["country"][country][year]["Biofuels and waste"] = energy_data["country"][country][year]["Renewables and waste"] - energy_data["country"][country][year]['Renewable sources']
            if year < 2000:
                energy_data["country"][country][year]["Hydro"] = energy_data["country"][country][year]['Renewable sources']
            else:
                energy_data["country"][country][year]["Hydro"] = energy_data["country"][country][2000][
                    'Renewable sources']
            energy_data["country"][country][year]['Wind, solar, etc.'] = energy_data["country"][country][year]['Renewable sources'] - energy_data["country"][country][year]["Hydro"]
            energy_data["country"][country][year]["exergy"] = np.sum([exergy_coef[ener]*energy_data["country"][country][year][ener] for ener in list(exergy_coef.keys())])
    energy_data = add_emissions(energy_data, country)
    return energy_data

def tfp(path="data/iea/tfp_countries.csv", country="France"):
    data = {country : []}
    with open(path, "r") as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            if country in row:
                index = row.index(country)
            else:
                data[country] += [float(row[index].replace(",","."))]
    return data

def get_scenario(scenario, path_exergy_var="scenario/var_exergy_", exergy_coefs_path = "data/iea/exergy_coefs.json", energy_path="data/iea/IEA_all_countries/countries_energy_balance.csv", country="United States"):


    assert country in ["France", "United States", "United Kingdom", "Germany", "Italy", "Canada", "Japan"]
    path_exergy_var = path_exergy_var + scenario
    assert os.path.exists(path_exergy_var)


    energy_dataset = countries_energy(exergy_coefs_path=exergy_coefs_path, energy_path=energy_path, country=country)
    with open(path_exergy_var, "r") as json_file:
        data_exergy_scenario = json.load(json_file)
    exergy_coef = get_exergy_coefs(exergy_coefs_path)
    sources = list(data_exergy_scenario[country]["2020"].keys())
    for year in range(2020, 2051):
        #2019 is the last observed year
        energy_dataset["country"][country].update({year : {source: energy_dataset["country"][country][year - 1][source]*(1 + data_exergy_scenario[country][str(year)][source]) for source in sources}})
        energy_dataset["country"][country][year]["exergy"] = np.sum(
            [exergy_coef[ener] * energy_dataset["country"][country][year][ener] for ener in sources])
        add_emissions(energy_dataset, country)

    return [energy_dataset["country"][country][year]["exergy"] for year in range(2020, 2051)], [energy_dataset["country"][country][year]["emissions"] for year in range(2020, 2051)]

def diff(data):
    return ([data[t] - data[t - 1] for t in range(1, len(data))])


def compute_tfp_level(variations):
    real_tfp = [1]
    for k in range(1, len(variations)):
        real_tfp += [real_tfp[k - 1] * (1 + variations[k] / 100)]
    log_tfp = [np.log(tfp) for tfp in real_tfp]
    return log_tfp


def get_exergy_iea(path="data/iea/exergy_from_iea_data.csv"):
    """
    Used to generate result of section 2
    :param path:
    :return:
    """
    exergy = []
    with open(path, "r") as csv_file:
        data = csv.reader(csv_file)
        for row in data:
            exergy += [float(row[0])]
    return exergy


def get_energy_iea(path="data/iea/energy.json"):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    data = [{"2020": {key: data[key] * 2.93 * 10e7 for key in list(data.keys())}}]
    return data


def get_exergy_coefs(path="data/iea/exergy_coefs.json"):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data


def get_exergy(energy, exergy_coefs):
    exergy = {}
    for k in range(len(energy)):
        print(energy[k].keys())
        year = list(energy[k].keys())[0]
        exergy.update({year: np.sum(
            [float(exergy_coefs[ener]) * float(energy[k][year][ener]) for ener in energy[0][year].keys()])})
    return exergy


def gen_exergy(efficiency, energy, scenario, horizon, c_e):
    pred_energy = [{key: energy[key] for key in energy.keys()} for _ in range(horizon)]
    for t in range(1, horizon):
        pred_energy[t].update(
            {key: pred_energy[t - 1][key] * (1 + float(scenario["year"][str(2021 + t - 1)][key])) for key in
             energy.keys()})
    with open("data/scenario_1.json", "w") as json_file:
        json.dump(pred_energy, json_file)
    exergy_scenario = [np.sum([efficiency[key] * pred_energy[t][key] for key in energy.keys()]) for t in range(horizon)]
    return diff([e / c_e for e in exergy_scenario])


def get_tfp(path="../data/tfp_data"):
    with open(path, "r") as txt_file:
        data = txt_file.readlines()

    return [float(value.replace("\n", "").replace(",", ".").replace('"', '')) for value in data]


def add_emissions(energie, country):

    years = list(energie["country"][country].keys())
    emissions_factor = {"Coal": 820, "Natural gas": 490, "Crude oil": 400, "Nuclear": 12, "Hydro": 24, "Biofuels and waste": 230, 'Wind, solar, etc.': 25}
    sources = list(emissions_factor.keys())

    for year in years:
        emissions = 0
        for source in sources:
            emissions += energie["country"][country][year][source]*emissions_factor[source]*11630/1000 #kg to tonnes
        energie["country"][country][year].update({"emissions": emissions})
    return energie

def weitzman(T, G):
    """

    :param T: Temperature
    :param G: Green house gas concentration
    :return: probability to get a high increase than T, according to : Weitzman, GHG targets as insurance against catastrophic climate damages.
    """

    phi = np.log(G / 280) / np.log(2)
    S = T/phi
    return norm.pdf(S, 3, 1.447)/phi




if __name__ == "__main__":
    step = 0.01
    T = np.arange(6, 40, step)
    Gs = [400 + k for k in range(0, 350, 50)]

    for G in Gs:
        dens = []
        for t in T:
            dens += [weitzman(t,G)]
        print(G)
        print(np.sum([step*(dens[t+1]+dens[t])/2 for t in range(len(dens) - 1)]))
    country = "France"
    energie = countries_energy(exergy_coefs_path
     = "../data/iea/exergy_coefs.json", energy_path="../data/iea/IEA_all_countries/countries_energy_balance.csv", country=country)

    emissions = [energie["country"][country][year]["emissions"] for year in range(1971, 2019)]
    print(emissions)
    import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(emissions)
    #plt.show()
