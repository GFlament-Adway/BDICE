import numpy as np
from simulate import simulate_PD, simulate_rho
from VaR import get_rho





# set the true values of the model parameters for creating the data
real_pd = 0.05
real_rho = 0.2
alpha = 0.001

# set the "predictor variable"/abscissa
M = 1000
time_horizon = 120


xmin = 0.
xmax = 1
stepsize = (xmax - xmin) / time_horizon
x = np.arange(xmin, xmax, stepsize)


# define the model function
def straight_line(x, m, c):
    """
    A straight line model: y = m*x + c

    Args:
        x (list): a set of abscissa points at which the model is defined
        m (float): the gradient of the line
        c (float): the y-intercept of the line
    """

    return m * x + c

# create the data - the model plus Gaussian noise
rho =  [get_rho(simulate_rho(real_rho, M)) for _ in range(time_horizon)]
pd = [np.mean(simulate_PD(real_pd, M)) for _ in range(time_horizon)]
#pd = [pd_est + 0.0005 if pd_est == 0 else pd_est for pd_est in pd]