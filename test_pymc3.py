import pymc3 as pm
import numpy as np
import arviz as az
from theano import tensor as tt
import matplotlib.pyplot as plt

if __name__ == "__main__":
    az.rcParams["plot.matplotlib.show"] = True
    x = [
        -1.64934805, 0.52925273, 1.10100092, 0.38566793, -1.56768245,
        1.26195686, 0.92613986, -0.23942803, 0.33933045, 1.14390657,
        0.65466195, -1.36229805, -0.32393554, -0.23258941, 0.17688024,
        1.60774334, -0.22801156, 1.53008133, -1.31431042, -0.27699609, 0.3
    ]  # inputs
    y = [
        -3.67385666, 3.37543275, 6.25390538, 1.41569973, -2.08413872,
        6.71560158, 6.32344159, 2.40651236, 4.54217349, 6.25778739,
        4.98933806, -2.69713137, 1.45705571, -0.49772953, 1.50502898,
        7.27228263, 1.6267433, 6.43580518, -0.50291509, 0.65674682, 1.8
    ]  # outputs

    with pm.Model() as predictive_model:
        # define priors
        a = pm.Normal('slope', 0, 16)
        b = pm.Normal('intercept', 0, 16)
        s = pm.Exponential('error', 1)

        # predictions
        x_ = pm.Data('features', x)  # a data container, can be changed
        obs = pm.Normal('observation', a * x_ + b, s, observed=y)

        # use MCMC to sample
        trace = pm.sample(draws=100, tune=50, cores=4, chains=2, return_inferencedata=True)
        axes = az.plot_posterior(trace)

    x_new = np.linspace(-3, 3, 50)  # 50 input values between -3 and 3

    with predictive_model:
        pm.set_data({'features': x_new})
        posterior = pm.sample_posterior_predictive(trace)

    y_pred = posterior['observation']
    y_mean = y_pred.mean(axis=0)
    y_std = y_pred.std(axis=0)


    plt.figure(figsize=(16, 8))
    plt.scatter(x, y, c='k', zorder=10, label='Data')
    plt.plot(x_new, y_mean, label='Prediction Mean')
    plt.fill_between(x_new, y_mean - 3 * y_std, y_mean + 3 * y_std, alpha=0.33,
                     label='Uncertainty Interval ($\mu\pm3\sigma$)')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim(-14, 16)
    plt.legend(loc='upper left')
    plt.show()