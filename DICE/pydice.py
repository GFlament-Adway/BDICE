import json
import numpy as np


class DICE:
    def __init__(self, parameter=None, tfp=None):
        if tfp is not None:
            self.tfp_model = "endogenous"
        else:
            self.tfp_model = "exogenous"
        self.population_growth = "given"
        if tfp is not None:
            self.tfp_traj = tfp
        if parameter is None:
            self.parameters = {
                "year": 2010,
                "output": [],
                "population": [7838],
                "capital share": 0.3,
                "population growth": 0.007,
                "tfp growth": 0.0165,
                "tfp": [4.5],
                "capital": [160],
                "depreciation": 0.1,
                "saving_rate": 0.26,
                "temperature augmentation": [1.2],
                "damage from t increase": [],
                "temp increase coef": 0.0008,
                "exergy": [0],
                "sources": ["Coal", "Crude oil", "Oil products", "natural gas", "Nuclear", "Hydro", "Wind, solar, etc.",
                            "Biofuels and waste"],
                "estimates emissions factor": 1, #Estimation des facteurs d'émissions par source : 0, 1 ou 2
                "damage function": {
                    "exponent": 2,
                    "coef": [0, 0, 0.002136]
                },
                "exergy coefs": {
                    "Coal": 1.088,
                    "Crude oil": 1.07,
                    "Oil products": 1.07,
                    "natural gas": 1.04,
                    "Nuclear": 1,
                    "Hydro": 0.85,
                    "Wind, solar, etc.": 0.15,
                    "Biofuels and waste": 1.15
                },
                "primary energy emission factor": {
                    # See https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
                    # (t/ktoe)
                    "Coal": [740 * 11630000 / 1000000, 820 * 11630000 / 1000000, 910 * 11630000 / 1000000],
                    "Crude oil": [740 * 11630000 / 1000000, 820 * 11630000 / 1000000, 910 * 11630000 / 1000000],
                    # Les valeurs semblent similaires entre pétrole et charbon : https://www.eia.gov/tools/faqs/faq.php?id=74&t=11
                    "Oil products": [740 * 11630000 / 1000000, 820 * 11630000 / 1000000, 910 * 11630000 / 1000000],
                    # C'est peut-être une erreur celui-là ...
                    "natural gas": [410 * 11630000 / 1000000, 490 * 11630000 / 1000000, 650 * 11630000 / 1000000],
                    "Nuclear": [3.7 * 11630000 / 1000000, 12 * 11630000 / 1000000, 110 * 11630000 / 1000000],
                    "Hydro": [1 * 11630000 / 1000000, 24 * 11630000 / 1000000, 2200 * 11630000 / 1000000],
                    "Wind, solar, etc.": [7 * 11630000 / 1000000, 11 * 11630000 / 1000000, 56 * 11630000 / 1000000],
                    # I only took onshore wind ...
                    "Biofuels and waste": [130 * 11630000 / 1000000, 230 * 11630000 / 1000000, 420 * 11630000 / 1000000]
                },
                "primary energy sources": {
                    # See https://en.wikipedia.org/wiki/Life-cycle_greenhouse_gas_emissions_of_energy_sources
                    # (t/ktoe)
                    "Coal": [0],
                    "Crude oil": [0],
                    # Les valeurs semblent similaires entre pétrole et charbon : https://www.eia.gov/tools/faqs/faq.php?id=74&t=11
                    "Oil products": [0],
                    # C'est peut-être une erreur celui-là ...
                    "natural gas": [0],
                    "Nuclear": [0],
                    "Hydro": [0],
                    "Wind, solar, etc.": [0],
                    # I only took onshore wind ...
                    "Biofuels and waste": [0]
                }

            }

    def dump_parameters(self, path="parameters.json"):
        with open(path, "w") as json_file:
            json.dump(self.parameters, json_file)


    def __exergy(self):
        self.parameters["exergy"] += [np.sum(
            self.parameters["primary energy sources"][source][-1] * self.parameters["exergy coefs"][source] for source
            in self.parameters["sources"])]
    def __climate_module(self):
        emissions = np.sum([
            self.parameters["primary energy emission factor"][source][self.parameters["estimates emissions factor"]] * self.parameters["primary energy sources"][source][-1]
            for source in self.parameters["sources"]
        ])
        self.parameters["temperature augmentation"] += [self.parameters["temp increase coef"] * emissions]

    def __damage_function(self):
        self.parameters["damage from t increase"] += [np.sum([self.parameters["damage function"]["coef"][k]*self.parameters["temperature augmentation"][-1]**k for k in range(self.parameters["damage function"]["exponent"])])]

    def __population_growth(self):
        self.parameters["population"] += [self.parameters["population"][-1]*(1 + self.parameters["population growth"])]
        self.parameters["population growth"] = max(0, self.parameters["population growth"] - (self.parameters["year"] - 2010)*0.000001)

    def __tfp_growth(self):
        self.parameters["tfp"] += [self.parameters["tfp"][-1]*(1 + self.parameters["tfp growth"])]
        self.parameters["tfp growth"] = self.parameters["tfp growth"]*(1-0.006)

    def __capital_growth(self):
        self.parameters["capital"] += [self.parameters["capital"][-1]*(1-self.parameters["depreciation"]) + self.parameters["output"][-1]*self.parameters["saving_rate"]]

    def __economic_module(self):
        population = self.parameters["population"][-1]/1000
        capital = self.parameters["capital"][-1]
        damage = self.parameters["damage from t increase"][-1]
        capital_share = self.parameters["capital share"]
        tfp = self.parameters["tfp"][-1]
        self.parameters["output"] += [tfp*population**(1-capital_share)*capital**capital_share*(1-damage)]


    def step(self):
        self.parameters["year"] += 1
        self.__climate_module()
        self.__damage_function()
        self.__economic_module()
        self.__population_growth()
        if self.tfp_model == "endogenous":
            self.parameters["tfp"] += [self.parameters["tfp"][-1]*(1+self.tfp_traj[self.parameters["year"] - 2011])]
        else:
            self.__tfp_growth()
        self.__exergy()
        self.__capital_growth()

if __name__ == "__main__":
    dice = DICE()
    for _ in range(90):
        dice.step()
    print([dice.parameters["tfp"][k] for k in range(0, len(dice.parameters["output"]), 5)])
    print([dice.parameters["population"][k] for k in range(0, len(dice.parameters["output"]), 5)])
    print([dice.parameters["capital"][k] for k in range(0, len(dice.parameters["output"]), 5)])

    print([dice.parameters["output"][k] for k in range(0, len(dice.parameters["output"]), 5)])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(dice.parameters["output"])
    plt.show()