import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from utils.utils import get_tfp, get_exergy_iea, compute_tfp_level, diff
from scenario import scenario
import json
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib

def var(data):
    """

    :param data: Data to compute variation
    :return: Return 100 * (data[t] - data[t - 1]) / data[t - 1] for all t
    """
    return ([100 * (data[t] - data[t - 1]) / data[t - 1] for t in range(1, len(data))])


def first_result():
    """
    TODO : Add error control

    :return: Return the first regression result.
    """
    font = {'weight': 'bold',
            'size': 20}

    matplotlib.rc('font', **font)

    horizon = 100
    data_scenario = scenario.gen_scenario(horizon)

    with open("data/scenario.json", "w") as json_file:
        json.dump(data_scenario, json_file)

    Exergy_US = get_exergy_iea()
    c_e = Exergy_US[0]
    Exergy_US = [[1, np.log(e/c_e)] for e in Exergy_US]
    #Exergy_US = var(Exergy_US)
    tfp = get_tfp("data/tfp")[:-1]

    tfp_level = compute_tfp_level(tfp)
    tfp_level = [t for t in tfp_level]  # 1.782614 correspond à la valeur de référence de la TFP
    exergy_level = Exergy_US
    print(len(tfp_level), len(exergy_level))

    import pandas as pd

    mod = sm.OLS(tfp_level, exergy_level)
    reg = mod.fit()

    #Figure 1
    print(len(reg.resid))
    plt.figure()
    plt.plot(reg.resid)
    plt.xticks([k for k in range(0, len(reg.resid)+1, 5)], [str(1947 + k) for k in range(0, len(reg.resid)+1, 5)])
    plt.draw()


    print("tfp", adfuller(tfp_level))
    print("diff tfp", adfuller(diff(tfp)))
    print(adfuller(reg.resid))
    n_obs = 0

    df = pd.DataFrame(np.array([diff(tfp_level), diff(np.array(exergy_level).T[1])]).T, columns=["TFP", "exergy"])
    model = VAR(df)

    res = model.fit(1)
    y = diff(tfp_level)[:-1]
    var_y = np.sum([(y[t] - np.mean(y)) ** 2 for t in range(len(y))])
    # PIB_US = var(PIB_US)
    print(res.summary())
    print("R2 :", 1 - np.sum(res.resid["TFP"].values ** 2) / np.sum(var_y))
    print(sm.stats.acorr_ljungbox(res.resid["TFP"], lags=[1, 2, 3, 4, 5, 10], return_df=True))
    print(sm.stats.stattools.jarque_bera(res.resid["TFP"]))
    print(adfuller(res.resid["TFP"]))
    plt.figure()
    plt.plot([k for k in range(len(tfp_level[2:]))], res.fittedvalues["TFP"], label="fitted")
    plt.plot(diff(tfp_level[1:]), label="observed")
    plt.xticks([k for k in range(0, len(res.fittedvalues["TFP"]), 5)], [1948 +k for k in range(0, len(res.fittedvalues["TFP"]), 5)])
    plt.legend()
    plt.ylabel("log variation of TFP")
    plt.show()

    plt.figure()
    plt.plot([k for k in range(len(tfp_level[2:]))], res.resid["TFP"], label="residuals")
    plt.xticks([k for k in range(0, len(res.fittedvalues["TFP"]), 5)],
               [1948 + k for k in range(0, len(res.fittedvalues["TFP"]), 5)])
    plt.legend()
    plt.ylabel("residuals")
    plt.show()

if __name__ == "__main__":
    first_result()

