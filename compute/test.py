import matplotlib.pyplot as plt
import numpy as np

from VaR import get_rho, VaR, var_env
from utils.get_data import get_energie, get_stock_value, get_data_yahoo, get_temperature
from utils.utils_math import normalize, cor, ls, mean_diff_length, diff, centre_red, accroissement
from utils.dim_red import EM, pca
import yfinance as yf

from scipy.stats import norm


def get_sample_comp(tickers=["UG.PA"]):
    data = []
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(period="25y")
        values = [list(hist["High"])]
        data += [np.log(value) for value in values]
    return data

def test_secteur(secteur="cosmetique", mean_comp=True):
    data = get_stock_value("industrie_" + secteur)
    energie = get_energie()
    energie_test = [1.55 * ener for ener in get_energie("energie_pred")]
    plt.figure()
    plt.plot([k for k in range(len(energie))], energie, label="Energie consommée dans le monde EJ (BP)")
    plt.plot([30 + k for k in range(len(energie_test))], [ener for ener in energie_test],
             label="Energie consommée dans le monde EJ (MEDEAS)")
    plt.xticks(ticks=[k for k in range(len(energie) + len(energie_test) - 25)],
               labels=[1965 + k for k in range(len(energie) + len(energie_test) - 25)], rotation=90)
    plt.legend()
    plt.draw()
    energie = get_energie()[-len(data[0]):]
    energie_test = energie_test[15:]
    for i, serie in enumerate(data):
        data[i] = normalize(serie)
    correl = [cor(serie, energie) for serie in data]
    print("correlation : ", correl)
    if mean_comp:
        data = [[np.mean([data[i][t] for i in range(len(data))]) for t in range(len(data[0]))]]
    least_squares = [ls(serie, energie) for serie in data]
    plt.figure()
    if mean_comp:
        plt.plot(energie, data[0])
    else:
        for i in range(len(data)):
            plt.plot(energie, data[i])
    plt.xlabel("Energie (EJ)")
    plt.ylabel("Valeur de l'actif")
    plt.draw()
    med = []
    for j in range(len(least_squares[0])):
        med += [np.median([least_squares[i][j] for i in range(len(least_squares))])]
    print(med)
    plt.figure()
    for i in range(len(data)):
        plt.plot(data[i], color="red", label="Valeur moyenne d'une entreprise du secteur " + secteur)
        # plt.plot([np.sum([least_squares[i][j-1]*energie[t]**j for j in range(1, 1 + len(least_squares[i]))]) for t in range(len(energie))], color=colors[i%len(colors)], marker="o")
    plt.plot([np.sum([med[j - 1] * energie_test[t] ** j for j in range(1, 1 + len(least_squares[0]))]) for t in
              range(len(energie_test))], color="blue", marker="*",
             label="Trajectoire estimée de la valeur du secteur " + secteur)
    plt.xticks([k for k in range(len(energie_test))], [2010 + k for k in range(len(energie_test))], rotation="90")
    plt.legend()
    plt.show()


def portefeuille():
    energie = normalize(get_energie()[-10:])
    data = get_stock_value("exemple_portefeuille")
    data = [normalize(series) for series in data]
    correl = [cor(serie, energie) for serie in data]
    print(np.mean(correl))
    print(get_rho(np.array(data)))


def test_composite():
    energie = normalize(get_energie()[-10:])
    data = get_stock_value("agri")
    data = [normalize(serie) for serie in data]
    correl = [cor(energie[:len(serie)], serie) for serie in data]
    print(correl)


def PIB_energy():
    energie = get_energie()
    pib = get_energie("pib_mondial")[5:]
    energie = [e for e in energie][:-1]
    print(energie, pib)
    reg = ls(pib, energie, deg=2, const=True)
    plt.figure()
    plt.plot(energie, pib, label="PIB en fonction de l'énergie consommée")
    plt.plot([e for e in energie], reg["pred"], label="Prédiction régression polynomiale d'ordre 2")
    plt.xlabel("Energie en EJ (BP)")
    plt.ylabel(r"PIB $10^{12}$ \$ (Banque mondiale)")
    plt.legend()
    plt.show()


def test_secteur_yahoo(secteur="agro", deg=1, func="poly", const=False):
    data,var = get_data_yahoo(secteur, return_var=True)
    data = [normalize(serie) for serie in data]
    var = mean_diff_length(var)
    temp = get_temperature()
    reg = ls(var, temp[-len(var):], deg=2, const=True, func="poly")
    plt.figure()
    plt.plot(var)
    plt.plot(reg["pred"])
    plt.draw()
    energie = normalize(get_energie())
    regs = []

    for serie in data:
        regs += [ls(serie, energie[-len(serie):], deg=deg, func=func, const=const)]
    beta = [np.mean([reg["beta"][i] for reg in regs]) for i in range(len(regs[0]["beta"]))]
    print(beta)
    mean = mean_diff_length(data)
    mean_pred = mean_diff_length([reg["pred"] for reg in regs])
    print("mse mean {secteur}: ".format(secteur=secteur),
          np.mean([(mean_pred[t] - mean[t]) ** 2 for t in range(len(mean))]))
    plt.figure()
    for serie in data:
        plt.plot([2020 - len(serie) + k for k in range(len(serie))], serie, color="blue", alpha=0.1)
    plt.plot([2020 - len(mean_pred) + k for k in range(len(mean_pred))], mean_pred, color="red", marker="*",
             label="Prédiction")
    plt.plot([2020 - len(mean) + k for k in range(len(mean))], mean, color="blue",
             label="Valeur moyenne des plus grosses entreprises du secteur")

    plt.title(secteur)
    plt.legend()
    plt.draw()


def test_energie(plot=True, decc=0.025, time_horizon=30, delay=0):
    renew = 2100 * 10e9
    hydro = 4200 * 10e9
    nuc = 2796 * 10e9
    non_renew = 13865 * 10e6 * 11630 - renew + hydro + nuc
    acc = 0.1
    delay = delay
    dec_non_renew = decc
    acc_non_renew = 0.1
    acc_nuc = 0.05
    non_renew_times = [(non_renew * (1 - dec_non_renew) ** (t-delay)) if t > delay else (non_renew * (1 + acc_non_renew) ** (t-delay)) for t in range(time_horizon)]
    if plot:
        plt.figure()
        plt.plot(non_renew_times,
                 label="Energie carbonée (scénario BAU)", color="brown")
        plt.plot([(renew * (1 + acc) ** t) for t in range(time_horizon)], label="Solaire + Eolien (+10%/an)",
                 color="green")
        plt.plot([nuc*(1+acc_nuc)**t for t in range(time_horizon)], label="Nucléaire")
        plt.plot([hydro for t in range(time_horizon)], label="hydro", color="blue")
        plt.plot(
            [(non_renew_times[t] + renew * (1 + acc) ** t) + nuc*(1+acc_nuc)**t + hydro for t in
             range(time_horizon)],
            label="Energie dans le monde", color="red")
        plt.xlabel("Année")
        plt.ylabel("Energie en kwh")
        plt.legend()
        plt.show()
    else:
        return [(non_renew_times[t] + renew * (1 + acc) ** t) + nuc*(1+acc_nuc)**t + hydro for t in
                range(time_horizon)]


def compute_ener_model(w=None, betas=None, plot=True, deccs=0.025, delay=10):
    if w is None:
        w = [30, 10, 50]
    if betas is None:
        betas = [[-0.49, 1.287], [-0.126, 0.912], [-0.126, 0.912]]
    data = get_data_yahoo("portefeuille")
    len_min = min([len(data[i]) for i in range(len(data))])
    data = [data[i][-len_min:] for i in range(len(data))]
    energie = normalize(get_energie())[-len_min:]
    value = normalize([np.sum([w[i] * data[i][t] for i in range(len(w))]) for t in range(len(data[0]))])
    sigma = np.sqrt(np.var(value))
    plt.figure()
    for decc in deccs:
        energie_pred = test_energie(plot=False, time_horizon=len_min, decc=decc, delay=delay)
        energie_pred = normalize(energie_pred, energie_pred[0])
        print(energie_pred)
        pred = [[np.sum([betas[sect][i - 1] * energie_pred[t] ** i for i in range(1, len(betas[0]) + 1)]) for t in
                 range(len(energie_pred))] for sect in range(len(data))]
        value_pred = [np.sum([w[i] * pred[i][t] / np.sum(w) for i in range(len(w))]) for t in range(len(value))]
        d = 0.5*value[-1]
        if plot:
            plt.plot(value)
            plt.plot([f + len(value) - 1 for f in range(len(value_pred))], value_pred, label="Valeur du portefeuille")
        plt.plot([norm.cdf((d-value_pred[t])/(np.sqrt(t)*sigma)) for t in range(1, len(value_pred))], label="Probabilité de défaut sous le scénario baisse de {decc}% des énergies fossiles".format(decc=np.round(100*decc,1)))
        #plt.plot([VaR(norm.cdf((d-value_pred[t])/(np.sqrt(t)*sigma)), rho=0.2, t=t) for t in range(1, len(value_pred))], label="VaR, sous le scénario baisse de {decc}% des énergies fossiles".format(decc=np.round(100*decc,1)))
        plt.xticks([k for k in range(len(value_pred))], [2021 + k for k in range(1, len(value_pred))], rotation=90)
        plt.ylabel(r"$PD$")
        plt.xlabel("Année")
    plt.legend()
    plt.show()

def test_variation(sector="auto", deg=2, func="poly", const=False):
    energie = normalize(diff(get_energie()))
    data = get_data_yahoo(sector)
    regs = []
    for i, serie in enumerate(data):
        data[i] = normalize(diff(serie))
        regs += [ls(data[i], energie[-len(data[i]):], deg=5, func=func, const=const)]
    mean_pred = mean_diff_length([reg["pred"] for reg in regs])
    mean = mean_diff_length(data)
    plt.figure()
    plt.plot(mean)
    plt.plot(mean_pred)
    plt.draw()

def compute_var_ener_model(w=None, betas=None, plot=True, deccs=0.025, delay=10):
    if w is None:
        w = [30, 10, 50]
    if betas is None:
        betas = [[-0.49, 1.287], [-0.126, 0.912], [-0.126, 0.912]]
    data, var = get_data_yahoo("portefeuille", return_var=True)
    len_min = min([len(data[i]) for i in range(len(data))])
    data = [data[i][-len_min:] for i in range(len(data))]
    value = normalize([np.sum([w[i] * data[i][t] for i in range(len(w))]) for t in range(len(data[0]))])

    plt.figure()
    for decc in deccs:
        energie_pred = test_energie(plot=False, time_horizon=len_min, decc=decc, delay=delay)
        energie_pred = normalize(energie_pred, energie_pred[0])
        pred = [[np.sum([betas[sect][i - 1] * energie_pred[t] ** i for i in range(1, len(betas[0]) + 1)]) for t in
                 range(len(energie_pred))] for sect in range(len(data))]
        value_pred = [np.sum([w[i] * pred[i][t] / np.sum(w) for i in range(len(w))]) for t in range(len(value))]
        if plot:
            plt.plot(value)
            plt.plot([f + len(value) - 1 for f in range(len(value_pred))], value_pred, label="Valeur du portefeuille")
            plt.draw()
        plt.plot([var_env(value_pred[t], risks=[0.05 for _ in range(t)], treshold=0.5*value[-1]) for t in range(1, len(value_pred))], label="Probabilité de défaut sous le scénario baisse de {decc}% des énergies fossiles".format(decc=np.round(100*decc,1)))
        #plt.plot([VaR(norm.cdf((d-value_pred[t])/(np.sqrt(t)*sigma)), rho=0.2, t=t) for t in range(1, len(value_pred))], label="VaR, sous le scénario baisse de {decc}% des énergies fossiles".format(decc=np.round(100*decc,1)))
        plt.xticks([k for k in range(len(value_pred))], [2021 + k for k in range(1, len(value_pred))], rotation=90)
        plt.ylabel(r"$PD$")
        plt.xlabel("Année")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    """
    sectors = ["pharma", "auto", "lourde", "technologique", "cosmetique", "agro"]
    
    for sector in sectors:
        test_secteur_yahoo(sector, deg=2, func="poly", const=False)
    plt.show()
    
    test_energie(delay=0)
    compute_var_ener_model(plot=False, deccs=[-0.025, 0.07], delay=0)
    """
    values = get_sample_comp(["^DJI", "^IXIC", "^GSPC"])
    #, "AAPL", "MSFT", "PEP", "AMD", "CME", "ADP", "BAB", "TSM",  "CSCO"])
        #)
         #["AAPL", "MSFT", "PEP", "AMD", "TWTR"])
                             #, "AZN", "MDLZ", "CME", "ADP", "BAB", "TSM", "JNJ", "MA", "PG", "MRK", "VZ", "CSCO"])
    energie = get_energie()

    values = [diff(serie) for serie in values[1:]]
    values_year = [[serie[220*(k-1):220*k] for k in range(1, len(serie)//220)] for serie in values]
    quantiles_year = [[sorted(year)[int(0.01*len(year))] for year in serie]for serie in values_year]
    energie = accroissement(energie)[-len(quantiles_year[0]):]
    print(energie)
    temperatures = get_temperature()[-len(quantiles_year[0]):]
    plt.figure()
    plt.plot(np.array(quantiles_year).T, color="blue")

    quantiles_year = np.array(quantiles_year)
    regs = []
    for quantiles in quantiles_year:
        print("cov : ", np.cov(quantiles, energie))
        regs += [ls(quantiles, energie, deg=1)]
    for reg in regs:
        plt.plot(reg["pred"], color="red", marker="*")
    print("eqm : ", [reg["eqm"] for reg in regs])
    print("beta : ", [reg["beta"] for reg in regs])
    #values = [accroissement(serie) for serie in values]
    #pca(values)
    plt.show()

