# -*- coding: utf-8 -*-
"""
Script for Verifying Soil Texture Parameters

Created on Thu Aug 21 11:43:07 2025

@author: kbdon
"""

import numpy as np
import pandas as pd
import subprocess
import os
import io
from pathlib import Path
import sys
import shutil

import torch
from torch.quasirandom import SobolEngine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

from datetime import datetime, timedelta
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time

param_sol = [("SOL_BD(1).sol", 0.95, 1.25),
             ("SOL_BD(2).sol", 1.1, 1.35),
             ("SOL_BD(3).sol", 1.2, 1.45),
             ("SOL_BD(4).sol", 1.2, 1.55),
             ("SOL_BD(5).sol", 1.4, 1.9),
             ("SOL_BD(6).sol", 1.5, 2.0),
             ("SOL_AWC(1).sol", 0.1, 0.4),
             ("SOL_AWC(2).sol", 0.1, 0.4),
             ("SOL_AWC(3).sol", 0.1, 0.4),
             ("SOL_AWC(4).sol", 0.1, 0.4),
             ("SOL_AWC(5).sol", 0.1, 0.4),
             ("SOL_AWC(6).sol", 0.1, 0.4),
             ("CLAY(1).sol",20,40),
             ("CLAY(2).sol",20,40),
             ("CLAY(3).sol",20,40),
             ("CLAY(4).sol",20,40),
             ("CLAY(5).sol",20,40),
             ("CLAY(6).sol",20,40)]
             
df_param = pd.DataFrame(param_sol,columns=["parameter","LB","UB"])
LB = torch.tensor(df_param.iloc[:,1].tolist())
UB = torch.tensor(df_param.iloc[:,2].tolist())

# Specifying ground truth data:
df1 = pd.read_csv('obtileQmin.csv')
dates = pd.date_range(start=f"1/1/2020", periods = 1461).strftime("%m/%d/%Y")
ground_truth = torch.tensor(df1.iloc[:,1:14].to_numpy())

# Sampling, and checking if sample violates conditions. 
# TODO: If it does, then it needs to be resampled for analysis. We won't have this problem in a latent box for TuRBO
samples = 10000
sobol = SobolEngine(18, scramble=True)
sobol_samples = sobol.draw(samples).to(dtype=dtype, device=device)

BD = sobol_samples[:,0:6]*(UB[0:6]-LB[0:6]) + LB[0:6]
check = torch.zeros(samples)


for i in range(samples):
    checki = 0
    for j in range(BD.size(1)-1):
        if BD[i,j+1] < 0.9*BD[i,j]:
            checki += 1
    if checki > 0:
        check[i] = 1

mask = (check == 1)
sobol_clean = sobol_samples[~mask]

# check = torch.zeros(len(BD_clean))
# for i in range(len(BD_clean)):    
#     checki = 0
#     for k in range(BD_clean.size(1)-1):    
#         if BD_clean[i,k+1] < BD_clean[i,k]:
#             checki += 1
#     if checki > 0:
#         check[i] += 1

# mask = (check == 1)
# BD_cleaner = BD_clean[~mask]        

# mask1 = check == 1
# mask2 = check == 2
# count1 = torch.sum(mask1.int()) 
# count2 = torch.sum(mask2.int())
    
# # Removing samples that violate conditions:
# sobol_10perc = sobol_samples[~mask1]
# sobol_incr = sobol_samples[~mask2]

# BD_clean = sobol_incr[:,0:6]*(UB[0:6]-LB[0:6]) + LB[0:6]    


# Calculating Wilting Point, Field Capacity, and Saturation for appropriate layers:
    
wilting = torch.empty(len(sobol_clean),3)
field_cap = torch.empty(len(sobol_clean),3)
sat = torch.empty(len(sobol_clean),3)

BD = sobol_clean[:,0:6]*(UB[0:6]-LB[0:6]) + LB[0:6]
AWC = sobol_clean[:,6:12]*(UB[6:12]-LB[6:12]) + LB[6:12]
clay = sobol_clean[:,12:18]*(UB[12:18]-LB[12:18]) + LB[12:18]


for i in range(3):
    for j in range(len(sobol_clean)):
        wilting[j,i] = (0.4*clay[j,i]*BD[j,i])/100
        field_cap[j,i] = wilting[j,i] + AWC[j,i]
        sat[j,i] = 1 - (BD[j,i]/2.65)
        
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime  

start_tr_date = datetime.date(2020,1,1)
end_tr_date = datetime.date(2023,12,31)
all_dates = [start_tr_date + datetime.timedelta(days=i) for i in range((end_tr_date - start_tr_date).days + 1)]

wilting_mean = torch.mean(wilting,dim=0)
field_cap_mean = torch.mean(field_cap,dim=0)
sat_mean = torch.mean(sat,dim=0)

wilting_std = torch.std(wilting,dim=0)
field_cap_std = torch.std(field_cap,dim=0)
sat_std = torch.std(sat,dim=0)


# # VWC Plots:
fig, ax = plt.subplots(3,1, figsize=(24, 12))

ax[0].plot(all_dates, ground_truth[:,1],marker="x",linestyle="none",c='g')
ax[0].axhline(y=wilting_mean[0],color='red',xmin=0,xmax=1461)
ax[0].axhline(y=field_cap_mean[0],color='black',xmin=0,xmax=1461)
ax[0].axhline(y=sat_mean[0],color='green',xmin=0,xmax=1461)
ax[0].fill_between(all_dates, wilting_mean[0]-wilting_std[0], wilting_mean[0]+wilting_std[0], color='red', alpha=0.15)
ax[0].fill_between(all_dates, field_cap_mean[0]-field_cap_std[0], field_cap_mean[0]+field_cap_std[0], color='black', alpha=0.15)
ax[0].fill_between(all_dates, sat_mean[0]-sat_std[0], sat_mean[0]+sat_std[0], color='green', alpha=0.15)
ax[0].set_ylabel("VWC - 10 cm", fontsize = 16)
ax[0].set_xlabel("Date", fontsize = 16)
ax[0].set_xlim(start_tr_date, end_tr_date)
ax[0].tick_params(axis='both', labelsize=16)
ax[0].grid(True)


ax[1].plot(all_dates, ground_truth[:,2],marker="x",linestyle="none",c='g')
ax[1].axhline(y=wilting_mean[1],color='red',xmin=0,xmax=1461)
ax[1].axhline(y=field_cap_mean[1],color='black',xmin=0,xmax=1461)
ax[1].axhline(y=sat_mean[1],color='green',xmin=0,xmax=1461)
ax[1].fill_between(all_dates, wilting_mean[1]-wilting_std[1], wilting_mean[1]+wilting_std[1], color='red', alpha=0.15)
ax[1].fill_between(all_dates, field_cap_mean[1]-field_cap_std[1], field_cap_mean[1]+field_cap_std[1], color='black', alpha=0.15)
ax[1].fill_between(all_dates, sat_mean[1]-sat_std[1], sat_mean[1]+sat_std[1], color='green', alpha=0.15)
ax[1].set_ylabel("VWC - 20 cm", fontsize = 16)
ax[1].set_xlabel("Date", fontsize = 16)
ax[1].set_xlim(start_tr_date, end_tr_date)
ax[1].tick_params(axis='both', labelsize=16)
ax[1].grid(True)

ax[2].plot(all_dates, ground_truth[:,3],marker="x",linestyle="none",c='g')
ax[2].axhline(y=wilting_mean[2],color='red',xmin=0,xmax=1461)
ax[2].axhline(y=field_cap_mean[2],color='black',xmin=0,xmax=1461)
ax[2].axhline(y=sat_mean[2],color='green',xmin=0,xmax=1461)
ax[2].fill_between(all_dates, wilting_mean[2]-wilting_std[2], wilting_mean[2]+wilting_std[2], color='red', alpha=0.15)
ax[2].fill_between(all_dates, field_cap_mean[2]-field_cap_std[2], field_cap_mean[2]+field_cap_std[2], color='black', alpha=0.15)
ax[2].fill_between(all_dates, sat_mean[2]-sat_std[2], sat_mean[2]+sat_std[2], color='green', alpha=0.15)
ax[2].set_ylabel("VWC - 50 cm", fontsize = 16)
ax[2].set_xlabel("Date", fontsize = 16)
ax[2].set_xlim(start_tr_date, end_tr_date)
ax[2].tick_params(axis='both', labelsize=16)
ax[2].grid(True)

handles, labels = ax[0].get_legend_handles_labels()
line1, = ax[0].plot([], [], marker="x", linestyle="none", color='g')  # Ground Truth
line2 = plt.Line2D([], [], color='red')             # Wilting Point
line3 = plt.Line2D([], [], color='black')           # Field Capacity
line4 = plt.Line2D([], [], color='green')           # Saturation

fig.legend([line1, line2, line3, line4],
            ['Ground Truth', 'Wilting Point', 'Field Capacity', 'Saturation'],
            loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)

plt.tight_layout()
plt.show()       
        
        