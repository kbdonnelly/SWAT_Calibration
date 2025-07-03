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
    
       
    if method == ['RMSE']:
                
        output = torch.zeros(sensors.size(1))
        for i in range(sensors.size(1)):
            
            # Creating a mask to account for NaNs in ground truth data for calculation:
            mask = ~torch.isnan(ground_truth[:, i])
            valid_gt = ground_truth[mask, i]
            valid_pred = sensors[mask, i]

            # Take only first 80% of valid data: 
            cutoff = int(0.8 * len(valid_gt))
            valid_gt = valid_gt[:cutoff]
            valid_pred = valid_pred[:cutoff]
            
            output[i] = torch.sqrt(torch.mean((valid_pred - valid_gt)**2)) / torch.std(valid_gt)
            
            # output[i] = torch.sqrt(torch.sum(torch.square((sensors[:,i][mask[:,i]]-ground_truth[:,i][mask[:,i]])/len(ground_truth[:,i][mask[:,i]]))))/torch.std(ground_truth[:,i][mask[:,i]])
         
    else:
        print('Please enter an accepted objective function.')
        output = 0
    
    return output

if __name__== '__main__':
    
    run_type = ['Input'] # Types accepted: ['Rand','Sobol','Input']
    plotting = True # Option for turning plotting on/off
    
    
    if run_type == ['Rand']:
        
        theta_turbo  = torch.tensor(pd.read_csv('df_X_TuRBO.csv').to_numpy()[0])
        theta_rand = torch.rand(498,dim)
        theta_rand = torch.cat([theta_turbo.unsqueeze(0),theta_rand])
        
        output = torch.empty(theta_rand.shape[0])
        for i in range(theta_rand.shape[0]):
            output[i] = torch.sum(objective_function(theta_rand[i]))
            print(f"[{i}] Random interation complete.")
        
        df_X_Random =  pd.DataFrame(theta_rand)
        df_X_Random.to_csv('df_X_Random.csv', sep=',', index = False, encoding='utf-8')
        df_Y_Random =  pd.DataFrame(output)
        df_Y_Random.to_csv('df_Y_Random.csv', sep=',', index = False, encoding='utf-8')
        
        
    if run_type == ['Sobol']:
        
        theta = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(10)
        
        output = torch.empty(len(theta),13)
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
        sensor12 = torch.empty(len(theta),1461)
        sensor13 = torch.empty(len(theta),1461)
    
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
            sensor12[i] = sensors[:,11]
            sensor13[i] = sensors[:,12]
            print(f" [{i+1}] Sobol run complete.")
        
    if run_type == ['Input']:
               
        theta1  = torch.tensor(pd.read_csv('df_X_TuRBO_070225.csv').to_numpy())
        output1 = -torch.tensor(pd.read_csv('df_Y_TuRBO_070225.csv').to_numpy())
        theta1 = theta1[torch.argmin(output1)]
        output_accum1 = np.minimum.accumulate(output1).numpy().flatten()
        

    if plotting == True:

        fig, ax = plt.subplots(figsize=(8, 6))  
        plt.plot(output_accum1, marker="", lw=3,c='b')
        # plt.plot(output_accum2, marker="", lw=3,c='g') 
        # plt.plot(output_accum3, marker="", lw=3,c='r')
        plt.plot(output1,marker=".",linestyle="none",c='b',alpha=0.1)
        ax.set_yscale('log')
        plt.ylabel("Loss", fontsize = 16)
        plt.xlabel("Evaluations", fontsize = 16)
        plt.legend(['TuRBO','TuRBO_eval'],loc='upper right',fontsize=12) 
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim([10,60])
        plt.xlim([0,675])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
  

                  

