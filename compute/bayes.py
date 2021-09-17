from __future__ import print_function, division

import os
import sys
import numpy as np

import matplotlib as mpl
mpl.use("Agg") # force Matplotlib backend to Agg

# import PyJAGS
import pyjags

# import model and data
from createdata import *

# Create model code
line_code_jags = '''
model {{
    for (i in 1:N) {{
        rho[i] ~ dbeta(a_1, b_1) T(0.0000001,0.9999999)
        pd[i] ~ dbeta(a_2, b_2) T(0.0000001,0.99999999)
        VaR[i] <- pnorm((qnorm(pd[i], 0, 1) - sqrt(rho[i])*qnorm({alpha}, 0, 1))/sqrt(1-rho[i]), 0, 1)
    }}
    rho_est ~ dunif({rho_min}, {rho_max})     # Gaussian prior on rho
    pd_est ~ dunif({pd_min}, {pd_max}) 
      # Gaussian prior on pd
    
    varrho <- ((-{rho_min} + {rho_max})**2)/12
    varpd <-  ((-{pd_min} + {pd_max})**2)/12
    
    b_1 <- (1-rho_est) * (rho_est*(1-rho_est) - varrho)/varrho
    a_1 <- a_2*rho_est/(1-rho_est)
    
    b_2 <- (1-pd_est) * (pd_est*(1-pd_est) - varpd)/varpd
    a_2  <- b_2*pd_est*pd_est/(1-pd_est)
}}
'''
"""
rho_est ~ dunif({mrho}, {varrho})     # Gaussian prior on rho
    pd_est ~ dunif({mpd}, {varpd})   
"""
datadict = {'N': time_horizon,    # number of data points
            'rho': rho,
            "pd": pd} # the observed data

Nsamples = 1000 # set the number of iterations of the sampler
chains = 4      # set the number of chains to run with

# dictionary for inputs into line_code
linedict = {}
linedict['pd_min'] = 0.001           # mean of Gaussian prior distribution for m
linedict["pd_max"] = 0.1
linedict["rho_min"] = 0.01
linedict["rho_max"] = 0.5
linedict["alpha"] = alpha
# compile model

model = pyjags.Model(line_code_jags.format(**linedict), data=datadict, chains=chains)
samples = model.sample(Nsamples, vars=['VaR', 'pd', 'rho']) # perform sampling

varchainjags = samples['VaR'].flatten()
pdchainjags = samples['pd'].flatten()
rhochainjags = samples['rho'].flatten()

# extract the samples
postsamples = np.vstack((varchainjags, pdchainjags, rhochainjags)).T

# plot posterior samples (if corner.py is installed)
try:
    print('hello')
    import corner # import corner.py
except ImportError:
    print('oupsy')
    sys.exit(1)

print('Number of posterior samples is {}'.format(postsamples.shape[0]))
import matplotlib.pyplot as plt
print(sorted(varchainjags)[int((1-alpha)*len(varchainjags))])
plt.hist(varchainjags)
plt.axvline(sorted(varchainjags)[int((1-alpha)*len(varchainjags))], label=r"Correct $VaR$")
plt.legend()
plt.savefig("mygraph.png")