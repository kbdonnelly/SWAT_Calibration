# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 14:49:17 2025

@author: kbdon
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from calib_objective import ObjFunc

from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

###############################################################################
# Importing Necessary Data
###############################################################################

# TuRBO Runs, No Texture Parameters:
    
theta1 = torch.tensor(pd.read_csv('df_theta_TuRBO1_IWTDN1_SWATFT0.csv').to_numpy())
output1 = torch.tensor(pd.read_csv('df_output_TuRBO1_IWTDN1_SWATFT0.csv').to_numpy())
theta_best1 = theta1[torch.argmin(output1)].to(dtype=torch.float32).unsqueeze(1)
output_minaccum1 = np.minimum.accumulate(output1.numpy())

theta2 = torch.tensor(pd.read_csv('df_theta_TuRBO1_IWTDN1_SWATFT1.csv').to_numpy())
output2 = torch.tensor(pd.read_csv('df_output_TuRBO1_IWTDN1_SWATFT1.csv').to_numpy())
theta_best2 = theta2[torch.argmin(output2)].to(dtype=torch.float32).unsqueeze(1)
output_minaccum2 = np.minimum.accumulate(output2.numpy())

theta3 = torch.tensor(pd.read_csv('df_theta_TuRBO1_IWTDN2_SWATFT0.csv').to_numpy())
output3 = torch.tensor(pd.read_csv('df_output_TuRBO1_IWTDN2_SWATFT0.csv').to_numpy())
theta_best3 = theta3[torch.argmin(output3)].to(dtype=torch.float32).unsqueeze(1)
output_minaccum3 = np.minimum.accumulate(output3.numpy())

theta4 = torch.tensor(pd.read_csv('df_theta_TuRBO1_IWTDN2_SWATFT1.csv').to_numpy())
output4 = torch.tensor(pd.read_csv('df_output_TuRBO1_IWTDN2_SWATFT1.csv').to_numpy())
theta_best4 = theta4[torch.argmin(output4)].to(dtype=torch.float32).unsqueeze(1)
output_minaccum4 = np.minimum.accumulate(output4.numpy())

###############################################################################
# Selecting Run to Test and Create GP
###############################################################################

seed = 0 # For replicates
dim = 15 # number of parameters
a = ObjFunc()

# param_list = [theta1, theta2, theta3, theta4]
# output_list = [output1, output2, output3, output4]

# Selecting parameters for case study of interest:
param = theta1
obj = output1
best_found = param[torch.argmin(obj)]

best_found_scaled = ((best_found - a.LB)/(a.UB-a.LB)).numpy()


# Specify the hypercube where we generate sobol samples:
    
lb_SA = best_found_scaled - 0.05 # we can also play around with the size of hypercube
ub_SA = best_found_scaled + 0.05

lb_SA = np.clip(lb_SA, 0, 1)
ub_SA = np.clip(ub_SA, 0, 1)

Ninit = 500 # number of data point generated around the best found parameter

train_X = torch.tensor(lb_SA) + torch.tensor(ub_SA - lb_SA)*torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit) # generate training data to train local GP
train_Y = []

for train_x in train_X:
    simulation_output = torch.sum(a(train_x,rescaled=False)) # run the simulation
    train_Y.append(simulation_output) 
    
train_Y = torch.stack(train_Y)

# Build model list:
    
model_list = []
# for nx in range(1):
#     covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
#     model_list.append(SingleTaskGP(train_X.to(torch.float64), train_Y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
model_list.append(SingleTaskGP(train_X.to(torch.float64), train_Y.unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))

model = ModelListGP(*model_list)
mll = SumMarginalLogLikelihood(model.likelihood, model)

# Fit the GPs
fit_gpytorch_mll(mll)

def objective(x):
    x = torch.tensor(x)
    output = model.posterior(x.unsqueeze(0).to(torch.float64)).mean.clone().detach()
    return output.squeeze(0).numpy()

from scipy.optimize import approx_fprime
x_opt = best_found_scaled  # your best parameter vector
epsilon = np.sqrt(np.finfo(float).eps)  # step size

grad = approx_fprime(x_opt, objective, epsilon)
print("Gradient:\n", grad)

from scipy.optimize._numdiff import approx_derivative

# Hessian approximation
hess = approx_derivative(lambda x: approx_fprime(x, objective, epsilon),
                         x_opt, method='2-point')
print("Hessian:\n", hess)

sym_error = np.linalg.norm(hess - hess.T)
print("Symmetry check:", sym_error)

eigvals = np.linalg.eigvals(hess)
print("Eigenvalues:", eigvals)

