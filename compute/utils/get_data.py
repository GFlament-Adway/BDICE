import os
import csv
import numpy as np


def get_stock_value(indus="industrie_pharma"):
    path = os.path.join(os.getcwd(), "../data/{indus}".format(indus=indus))
    assert os.path.exists(
        os.path.join(os.getcwd(), "../data/{indus}".format(indus=indus))), "Mauvais chemin : {path}".format(path=path)
    files = os.listdir(os.path.join(os.getcwd(), "../data/{indus}".format(indus=indus)))
    dict_mean_values = {}
    data = [[] for _ in range(len(files))]
    for i, file in enumerate(files):
        with open(os.path.join(path, file)) as txt_file:
            lines = txt_file.readlines()
            for line in lines:
                try:
                    data_on_line = line.split("\t")
                    year = data_on_line[0].split("/")[-1]
                    if year in dict_mean_values.keys():
                        dict_mean_values[year] += [float(data_on_line[1])]
                    else:
                        dict_mean_values[year] = [float(data_on_line[1])]
                except:
                    pass
        data[i] = [float(np.mean(dict_mean_values[year])) for year in list(dict_mean_values.keys())]
    print([len(series) for series in data])
    return data


def get_energie(fn="energie"):
    file = os.path.join(os.getcwd(), "../data/{fn}".format(fn=fn))
    data = []
    with open(file) as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            data = [float(value) for value in line.split("\t")]
    return data


def get_data_yahoo(fn="agro", return_var=False):
    path = os.path.join(os.getcwd(), "../data/industrie_{fn}_yahoo".format(fn=fn))
    indus = "industrie_{fn}_yahoo".format(fn=fn)
    files = os.listdir(os.path.join(os.getcwd(), "../data/{indus}".format(indus=indus)))
    dict_mean_values = {}
    data = [[] for _ in range(len(files))]
    var = [[] for _ in range(len(files))]
    for i, file in enumerate(files):
        dict_mean_values = {}
        with open(os.path.join(path, file)) as csv_file:
            rows = csv.reader(csv_file)
            for row in rows:
                data_on_line = row
                year = data_on_line[0].split("-")[0]
                try:
                    if year in dict_mean_values.keys():
                        dict_mean_values[year] += [float(data_on_line[1])]
                    else:
                        dict_mean_values[year] = [float(data_on_line[1])]
                except:
                    pass
        dict_mean_values.pop("2021")
        data[i] = [float(np.mean(dict_mean_values[year])) for year in list(dict_mean_values.keys())]
        var[i] = [float(np.var(dict_mean_values[year])) for year in list(dict_mean_values.keys())]
    if return_var:
        return data, var
    return data

def get_indices(fn="CAC40.txt"):
    file = os.path.join(os.getcwd(), "../data/indices/{fn}".format(fn=fn))
    data = []
    with open(file) as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            try:
                data += [float(line.split("\t")[1])]
            except ValueError:
                pass
    return data


def get_biotime():
    data = []
    with open("../../data/BioTime.csv") as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            print(row)


def get_temperature():
    data = []
    with open("../data/data_temperature.csv") as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            data += [float(row[1].replace(",", "."))]
    return data
if __name__ == "__main__":
    data = get_temperature()
    print(data)