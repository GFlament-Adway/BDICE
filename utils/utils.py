import csv
import numpy as np
import json


def diff(data):
    return ([data[t] - data[t - 1] for t in range(1, len(data))])

def compute_tfp_level(variations):
    real_tfp = [1]
    for k in range(1, len(variations)):
        real_tfp += [real_tfp[k-1]*(1 + variations[k]/100)]
    return real_tfp


def get_data(path="data/energy.csv"):
    data = {}
    with open(path, "r") as csv_file:
        raw_data = csv.reader(csv_file, delimiter=";")
        for row in raw_data:
            try:
                data.update({row[0] : float(row[1].replace(",", "."))})
            except ValueError:
                print(row)
    return data

def get_exergy_iea(path="data/iea/exergy_from_iea_data.csv"):
    exergy = []
    with open(path, "r") as csv_file:
        data = csv.reader(csv_file)
        for row in data:
            exergy += [float(row[0])]
    return exergy

def get_energy_iea(path="data/iea/energy.json"):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    data = [{"2020" : {key : data[key]*2.93*10e7 for key in list(data.keys())}}]
    return data

def get_exergy_coefs(path ="data/iea/exergy_coefs.json"):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data

def get_exergy(energy, exergy_coefs):
    exergy = {}
    for k in range(len(energy)):
        year = list(energy[k].keys())[0]
        exergy.update({year : np.sum([float(exergy_coefs[ener])*float(energy[k][year][ener]) for ener in energy[0][year].keys()])})
    return exergy

def gen_exergy(efficiency, energy, scenario, horizon, c_e):
    pred_energy = [{key : energy[key] for key in energy.keys()} for _ in range(horizon)]
    for t in range(1, horizon):
        pred_energy[t].update({key : pred_energy[t-1][key]*(1+float(scenario["year"][str(2021 + t-1)][key])) for key in energy.keys()})
    with open("data/scenario_1.json", "w") as json_file:
        json.dump(pred_energy, json_file)
    exergy_scenario = [np.sum([efficiency[key]*pred_energy[t][key] for key in energy.keys()]) for t in range(horizon)]
    return diff([e / c_e for e in exergy_scenario])

def get_tfp(path="../data/tfp_data"):
    with open(path, "r") as txt_file:
        data = txt_file.readlines()

    return [float(value.replace("\n","").replace(",",".").replace('"','')) for value in data]

if __name__ == "__main__":
    exergy_iea = get_exergy_iea("../data/iea/exergy_from_iea_data.csv")

