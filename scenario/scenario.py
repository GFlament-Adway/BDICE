def growth_per_sources(scenari = "Paris"):
    if scenari == "Paris":
        return {"Coal": -0.04, "Natural gas": -0.04, "Crude oil": -0.04, "Nuclear": 0, "Hydro": 0.01,
         "Biofuels and waste": 0.01}
    return {"Coal": 0.02, "Natural gas": 0.02, "Crude oil": 0.02, "Nuclear": 0, "Hydro": 0.01,
            "Biofuels and waste": 0.01}

def gen_peak():
    return {"Coal" : 2030, "Crude oil" : 2030, "Natural gas": 2030}

def gen_scenario(horizon=100, scenari="Paris", peak = None):
    """
    :param horizon: Nombre de pas de temps à générer.
    :return:
    """
    if peak:
        peak = gen_peak()
    data_scenario = {"year": {str(2021 + k) : {} for k in range(horizon + 1)}}
    growth = growth_per_sources(scenari)
    for k in range(horizon + 1):
        if peak:
            for key in list(peak.keys()):
                if 2021 + k == peak[key]:
                    growth[key] = - growth[key]
        data_scenario["year"][str(2021 + k)].update(growth)
    return data_scenario

if __name__ == "__main__":
    print(gen_scenario())
