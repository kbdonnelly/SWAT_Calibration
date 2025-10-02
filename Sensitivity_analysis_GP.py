# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:47:27 2025

@author: tang.1856, donnelly.235
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
from calib_objective import ObjFunc
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import torch
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

seed = 0 # For replicates
dim = 15 # number of parameters

a = ObjFunc()

# Importing parameters for case study of interest:
df_param = pd.read_csv('df_theta_TuRBO1_IWTDN1_SWATFT0.csv')
param = torch.tensor(df_param.to_numpy())
df_obj = pd.read_csv('df_output_TuRBO1_IWTDN1_SWATFT0.csv')
obj = torch.tensor(df_obj.to_numpy())

best_found = param[torch.argmin(obj)].numpy()
 
# Specify the hypercube where we generate sobol samples:
    
lb_SA = best_found - 0.05 # we can also play around with the size of hypercube
ub_SA = best_found + 0.05

np.clip(lb_SA, 0, 1)
np.clip(ub_SA, 0, 1)

Ninit = 500 # number of data point generated around the best found parameter

train_X = torch.tensor(lb_SA) + torch.tensor(ub_SA - lb_SA)*torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit) # generate training data to train local GP
train_Y = []

for train_x in train_X:
    simulation_output = a(train_x) # run the simulation
    train_Y.append(simulation_output) 
    
train_Y = torch.stack(train_Y)

# Build model list:
    
model_list = []
for nx in range(8):
    covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
    model_list.append(SingleTaskGP(train_X.to(torch.float64), train_Y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
model = ModelListGP(*model_list)
mll = SumMarginalLogLikelihood(model.likelihood, model)

# Fit the GPs
fit_gpytorch_mll(mll)


problem = {
    'num_vars':dim,
    'names':['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'],
    'bounds':[[lb_SA[0], ub_SA[0]],
              [lb_SA[1], ub_SA[1]],
              [lb_SA[2], ub_SA[2]],
              [lb_SA[3], ub_SA[3]],
              [lb_SA[4], ub_SA[4]],
              [lb_SA[5], ub_SA[5]],
              [lb_SA[6], ub_SA[6]],
              [lb_SA[7], ub_SA[7]],
              [lb_SA[8], ub_SA[8]],
              [lb_SA[9], ub_SA[9]],
              [lb_SA[10], ub_SA[10]],
              [lb_SA[11], ub_SA[11]],
              [lb_SA[12], ub_SA[12]],
              [lb_SA[13], ub_SA[13]],
              [lb_SA[14], ub_SA[14]]]   
    }

param_values = saltelli.sample(problem, 1024)

Y_0 = np.zeros([param_values.shape[0]])
Y_1 = np.zeros([param_values.shape[0]])
Y_2 = np.zeros([param_values.shape[0]])
Y_3 = np.zeros([param_values.shape[0]])
Y_4 = np.zeros([param_values.shape[0]])
Y_5 = np.zeros([param_values.shape[0]])
Y_6 = np.zeros([param_values.shape[0]])
Y_7 = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    # simulator_output = simulator_run(torch.tensor(X)) # run simulation
    GP_mean = model.posterior(torch.tensor(X).unsqueeze(0).to(torch.float64)).mean.flatten() # instead of performing true simulation, we estimate the output with GP posterior mean
    
    Y_0[i] = float(GP_mean[0])
    Y_1[i] = float(GP_mean[1])
    Y_2[i] = float(GP_mean[2])
    Y_3[i] = float(GP_mean[3])
    Y_4[i] = float(GP_mean[4])
    Y_5[i] = float(GP_mean[5])
    Y_6[i] = float(GP_mean[6])
    Y_7[i] = float(GP_mean[7])
   
    
Si_sensor1 = sobol.analyze(problem, Y_0)
Si_sensor2 = sobol.analyze(problem, Y_1)
Si_sensor3 = sobol.analyze(problem, Y_2)
Si_sensor4 = sobol.analyze(problem, Y_3)
Si_sensor5 = sobol.analyze(problem, Y_4)
Si_sensor6 = sobol.analyze(problem, Y_5)
Si_sensor7 = sobol.analyze(problem, Y_6)
Si_sensor8 = sobol.analyze(problem, Y_7)


print(Si_sensor1['S1'])
print(Si_sensor2['S1'])
print(Si_sensor3['S1'])
print(Si_sensor4['S1'])
print(Si_sensor5['S1'])
print(Si_sensor6['S1'])
print(Si_sensor7['S1'])
print(Si_sensor8['S1'])



import matplotlib.pyplot as plt
 
# Clean (clip negative and >1)
def clean_si(si):
    return np.clip(si, 0, 1)
 
S1 = map(clean_si, [Si_sensor1['S1']])
S2 = map(clean_si, [Si_sensor2['S1']])
S3 = map(clean_si, [Si_sensor3['S1']])
S4 = map(clean_si, [Si_sensor4['S1']])
S5 = map(clean_si, [Si_sensor5['S1']])
S6 = map(clean_si, [Si_sensor6['S1']])
S7 = map(clean_si, [Si_sensor7['S1']])
S8 = map(clean_si, [Si_sensor8['S1']])

# #Stack all results
# # all_S = [Si_sensor1['S1']]
# output_labels = "TuRBO-1: No Improvements"
# # param_labels = a.params
# param_labels = ['SFTMP.bsn',
#                 'SMTMP.bsn',
#                 'TIMP.bsn',
#                 'SMFMX.bsn',
#                 'SMFMN.bsn',
#                 'SNOCOVMX.bsn',
#                 'SNO50COV.bsn',
#                 'ESCO.hru',
#                 'EPCO.hru',
#                 'R2ADJ.hru',
#                 'OV_N.hru',
#                 'DEP_IMP.hru',
#                 'CN2.mgt',
#                 'LATKSATF.sdr',       
#                 'SOL_CRK.sol']


# # Plot: one subplot per output
# fig, ax = plt.subplots(figsize=(10, 10))
# si = Si_sensor1['S1']
# ax.barh(np.arange(15), si, color='tab:blue')
# ax.set_title(output_labels,fontsize=16)
# ax.set_xlabel("Si",fontsize=16)
# ax.tick_params(axis='x', labelsize=16)
# ax.set_xlim(0, 1)  # Set x-axis range 0 to 1
 
#     # Only show parameter labels on the leftmost plot
#     # if i == 0:
#     #     ax.set_yticks(np.arange(len(param_labels)))
#     #     ax.set_yticklabels(param_labels)
#     # else:
#     #     ax.set_yticks(np.arange(len(param_labels)))
#     #     ax.set_yticklabels([])
# ax.set_yticks(np.arange(len(param_labels)))
# ax.set_yticklabels(param_labels,fontsize=16)
# #plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.grid(True)
# plt.show()