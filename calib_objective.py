#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Objective for Environmental Model Calibration
@author: donnelly.235
"""
import sys
import torch
from torch import Tensor
from SWATrun import SWATrun
from scipy.stats import qmc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.double)

seed = 0
simulator = SWATrun()
dim = simulator.theta_dim

    
def objective_function(theta, method = ['RMSE']):  
    
    # Rescaling:
    LB = simulator.LB
    UB = simulator.UB
    
    theta_scaled = LB + (UB - LB)*theta
    
    # Running model to obtain desired outputs:    
    sensors = simulator.model_run(theta_scaled)   
    
    # Obtaining ground truth data from simulator:
    ground_truth = simulator.ground_truth
    
    
    # Creating a mask to account for NaNs in ground truth data for calculation:
    mask = ~torch.isnan(ground_truth)
       
    if method == ['RMSE']:
        
        # train_split = 0.8 # Use this for treating 80% of the output for testing
        
        output = torch.zeros(sensors.size(1))
        for i in range(sensors.size(1)):
            output[i] = torch.sqrt(torch.sum(torch.square((sensors[:,i][mask[:,i]]-ground_truth[:,i][mask[:,i]])/len(ground_truth[:,i][mask[:,i]]))))/torch.std(ground_truth[:,i][mask[:,i]])
         
    else:
        print('Please enter an accepted objective function.')
        output = 0
    
    return output

if __name__== '__main__':
    
    run_type = ['Sobol'] # Types accepted: ['Rand','Sobol','Input']
    plotting = False # Option for turning plotting on/off
    
    
    if run_type == ['Rand']:
        
        theta = torch.rand(dim)
        output = objective_function(theta)
        
    if run_type == ['Sobol']:
        
        theta = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(10)
        
        output = torch.empty(len(theta),11)
        sensor1 = torch.empty(len(theta),1461)
        sensor2 = torch.empty(len(theta),1461)
        sensor3 = torch.empty(len(theta),1461)
        sensor4 = torch.empty(len(theta),1461)
        sensor5 = torch.empty(len(theta),1461)
        sensor6 = torch.empty(len(theta),1461)
        sensor7 = torch.empty(len(theta),1461)
        sensor8 = torch.empty(len(theta),1461)
        sensor9 = torch.empty(len(theta),1461)
        sensor10 = torch.empty(len(theta),1461)
        sensor11 = torch.empty(len(theta),1461)
        # sensor12 = torch.empty(len(theta),1461)
        # sensor13 = torch.empty(len(theta),1461)
    
        for i in range(len(theta)):
            output[i], sensors = objective_function(theta[i],['RMSE'])
            sensor1[i] = sensors[:,0]
            sensor2[i] = sensors[:,1]
            sensor3[i] = sensors[:,2]
            sensor4[i] = sensors[:,3]
            sensor5[i] = sensors[:,4]
            sensor6[i] = sensors[:,5]
            sensor7[i] = sensors[:,6]
            sensor8[i] = sensors[:,7]
            sensor9[i] = sensors[:,8]
            sensor10[i] = sensors[:,9]
            sensor11[i] = sensors[:,10]
            # sensor12[i] = sensors[:,11]
            # sensor13[i] = sensors[:,12]
            print(f" [{i+1}] Sobol run complete.")
        
    if run_type == ['Input']:
               
        theta  = torch.tensor(pd.read_csv('CTurbo_test.csv').to_numpy())
        output = torch.empty(len(theta),12)
        for i in range(len(theta)):
            output[i] = objective_function(theta[i],['RMSE'])
        
        theta_best = theta[torch.argmin(torch.sum(output,dim=1))]
        output_accum = np.minimum.accumulate(torch.sum(output,dim=1).numpy().flatten())

    if plotting == True:

        fig, ax = plt.subplots(figsize=(8, 6))  
        plt.plot(output_accum, marker="", lw=3,c='b')    
        plt.plot(torch.sum(output,dim=1).numpy().flatten(),marker=".",linestyle="none",c='b',alpha=0.1)
        ax.set_yscale('log')
        plt.ylabel("Loss", fontsize = 16)
        plt.xlabel("Evaluations", fontsize = 16)
        plt.legend(['Sobol_min','Sobol_eval'],loc='upper right',fontsize=12) 
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim([0.1,1])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
  

                  

