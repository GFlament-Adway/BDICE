import pymc3 as pm
import numpy as np
import arviz as az
import scipy.stats as stats


def from_posterior(param, samples, k=100):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, k)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return pm.Interpolated(param, x, y)



if __name__ == "__main__":
    Y1 = predict(-0.04, horizon=1, sample_size=1)
    with pm.Model() as model1:
        alpha_0 = from_posterior(r"$\alpha_0$", trace[r"$\alpha_0"])
        likelihood = pm.Normal("y",
                               mu=intercept[country_id] + x_coeff[country_id] * x + x_ar[country_id] * x_exergy
                                  + mu_x_us[country_id] * x_us,
                               sigma=sigma[country_id],
                               observed=y)
