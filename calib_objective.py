#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Objective for Environmental Model Calibration
@author: kbdon

Last updated: 08/14/2025
"""
import sys
import torch
from torch import Tensor
from SWATrun import SWATrun
from scipy.stats import qmc
import pandas as pd
import numpy as np
from turbo_1 import Turbo1

import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.double)

class ObjFunc:
    def __init__(self):
        self.simulator = SWATrun()
        self.dim = self.simulator.theta_dim

        self.LB = self.simulator.LB
        self.UB = self.simulator.UB
        
    def __call__(self,theta):  
        
        # theta_scaled = LB + (UB - LB)*theta
        
        # Running model to obtain desired outputs:    
        sensors = self.simulator(theta)
        sensors_cali = sensors[639:-1,:]
        
        # Obtaining ground truth data from simulator:
        ground_truth = self.simulator.ground_truth[:,0:8]
        cali_range = ground_truth[639:-1,:]
        vali_range = ground_truth[0:639,:]
         
        output = torch.zeros(sensors.size(1))
        
        for i in range(sensors.size(1)):
            
            # Creating a mask to account for NaNs in ground truth data for calculation:
            mask = ~torch.isnan(cali_range[:, i])
            valid_gt = cali_range[mask, i]
            valid_pred = sensors_cali[mask, i]

            # # Take only first 80% of valid data: 
            # cutoff = int(0.8 * len(valid_gt))
            # valid_gt = valid_gt[:cutoff]
            # valid_pred = valid_pred[:cutoff]
            
            output[i] = torch.sqrt(torch.mean((valid_pred - valid_gt)**2)) / torch.std(valid_gt)
            
            # output[i] = torch.sqrt(torch.sum(torch.square((sensors[:,i][mask[:,i]]-ground_truth[:,i][mask[:,i]])/len(ground_truth[:,i][mask[:,i]]))))/torch.std(ground_truth[:,i][mask[:,i]])
             
        return output

if __name__== '__main__':
    
    simulator = SWATrun()
    f = ObjFunc()
    dim = simulator.theta_dim
    run_type = ['TuRBO-1'] # Types accepted: ['Rand','Sobol','Input','TuRBO-1']
    plotting = False # Option for turning plotting on/off
    seed = 0    
    
    if run_type == ['Rand']:
        
        theta = torch.rand(1,dim)
        LB = simulator.LB
        UB = simulator.UB
        
        theta_scaled = LB + (UB - LB)*theta
        output = f(theta_scaled.squeeze(0))
        
        # for i in range(theta_scaled.shape[0]):
        #     output[i] = ObjFunc(theta_scaled[i])
        #     print(f"[{i}] Random interation complete.")
        
        # df_X_Random =  pd.DataFrame(theta_rand)
        # df_X_Random.to_csv('df_X_Random.csv', sep=',', index = False, encoding='utf-8')
        # df_Y_Random =  pd.DataFrame(output)
        # df_Y_Random.to_csv('df_Y_Random.csv', sep=',', index = False, encoding='utf-8')
        
        
    if run_type == ['Sobol']:
        
        theta = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(10)
        
        # output = torch.empty(len(theta),13)
        # sensor1 = torch.empty(len(theta),1461)
        # sensor2 = torch.empty(len(theta),1461)
        # sensor3 = torch.empty(len(theta),1461)
        # sensor4 = torch.empty(len(theta),1461)
        # sensor5 = torch.empty(len(theta),1461)
        # sensor6 = torch.empty(len(theta),1461)
        # sensor7 = torch.empty(len(theta),1461)
        # sensor8 = torch.empty(len(theta),1461)
        # sensor9 = torch.empty(len(theta),1461)
        # sensor10 = torch.empty(len(theta),1461)
        # sensor11 = torch.empty(len(theta),1461)
        # sensor12 = torch.empty(len(theta),1461)
        # sensor13 = torch.empty(len(theta),1461)
    
        # for i in range(len(theta)):
        #     output[i], sensors = objective_function(theta[i],['RMSE'])
        #     sensor1[i] = sensors[:,0]
        #     sensor2[i] = sensors[:,1]
        #     sensor3[i] = sensors[:,2]
        #     sensor4[i] = sensors[:,3]
        #     sensor5[i] = sensors[:,4]
        #     sensor6[i] = sensors[:,5]
        #     sensor7[i] = sensors[:,6]
        #     sensor8[i] = sensors[:,7]
        #     sensor9[i] = sensors[:,8]
        #     sensor10[i] = sensors[:,9]
        #     sensor11[i] = sensors[:,10]
        #     sensor12[i] = sensors[:,11]
        #     sensor13[i] = sensors[:,12]
        #     print(f" [{i+1}] Sobol run complete.")
        
    if run_type == ['Input']:
        
        theta1 = torch.tensor(pd.read_csv('df_theta_TuRBO1_IWTDN1_SWATFT0.csv').to_numpy())
        output1 = torch.tensor(pd.read_csv('df_output_TuRBO1_IWTDN1_SWATFT0.csv').to_numpy())
        theta1_best = theta1[torch.argmin(output1)].to(dtype=torch.float32).unsqueeze(1)
        output1_minaccum = np.minimum.accumulate(output1.numpy())
        
        theta2 = torch.tensor(pd.read_csv('df_theta_TuRBO1_IWTDN1_SWATFT1.csv').to_numpy())
        output2 = torch.tensor(pd.read_csv('df_output_TuRBO1_IWTDN1_SWATFT1.csv').to_numpy())
        theta2_best = theta2[torch.argmin(output2)].to(dtype=torch.float32).unsqueeze(1)
        output2_minaccum = np.minimum.accumulate(output2.numpy())
        
        theta3 = torch.tensor(pd.read_csv('df_theta_TuRBO1_IWTDN2_SWATFT1.csv').to_numpy())
        output3 = torch.tensor(pd.read_csv('df_output_TuRBO1_IWTDN2_SWATFT1.csv').to_numpy())
        theta3_best = theta3[torch.argmin(output3)].to(dtype=torch.float32).unsqueeze(1)
        output3_minaccum = np.minimum.accumulate(output3.numpy())
        
        
        
        
        
    if run_type == ['TuRBO-1']:
               
        f = ObjFunc()
        turbo1 = Turbo1(
             f = f,  # Handle to objective function
             lb = np.array(f.LB),  # Numpy array specifying lower bounds
             ub = np.array(f.UB),  # Numpy array specifying upper bounds
             n_init = 2*dim,  # Number of initial bounds from an Latin hypercube design
             max_evals = 2000,  # Maximum number of evaluations
             batch_size = 10,  # How large batch size TuRBO uses
             verbose = True,  # Print information from each batch
             use_ard = True,  # Set to true if you want to use ARD for the GP kernel
             max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
             n_training_steps = 50,  # Number of steps of ADAM to learn the hypers
             min_cuda = 1024,  # Run on the CPU for small datasets
             device = "cpu",  # "cpu" or "cuda"
             dtype = "float64",  # float64 or float32
             seed=seed
         )
        turbo1.optimize()
        
        X = turbo1.X  # Evaluated points
        fX = turbo1.fX  # Observed values
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]
        
        print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
        
        df_theta_TuRBO1 =  pd.DataFrame(X)
        df_theta_TuRBO1.to_csv('df_theta_TuRBO1_IWTDN0_SWATFT0.csv', sep=',', index = False, encoding='utf-8')
        
        df_output_TuRBO1 =  pd.DataFrame(fX)
        df_output_TuRBO1.to_csv('df_output_TuRBO1_IWTDN0_SWATFT0.csv', sep=',', index = False, encoding='utf-8')

    if plotting == True:

        fig, ax = plt.subplots(figsize=(8, 6))  
        plt.plot(output1_minaccum, marker="", lw=3,c='orange')
        plt.plot(output2_minaccum, marker="", lw=3,c='b')
        plt.plot(output3_minaccum, marker="", lw=3,c='r')
        #plt.plot(fX,marker=".",linestyle="none",c='b',alpha=0.1)
        ax.set_yscale('log')
        plt.ylabel("NRMSE", fontsize = 16)
        plt.xlabel("Evaluations", fontsize = 16)
        plt.legend(['TuRBO-1: No Improvements','TuRBO-1: TMP','TuRBO-1: WTBL & TMP'],loc='upper right',fontsize=12) 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(1,3, figsize=(16, 6))  
        ax[0].plot(output1_minaccum, marker="", lw=3,c='orange')
        ax[0].plot(output1, marker=".",linestyle="none",c='orange',alpha=0.1)
        ax[0].set_yscale('log')
        ax[0].set_ylabel("NRMSE", fontsize = 16)
        ax[0].set_xlabel("Evaluations", fontsize = 16)
        ax[0].legend(['TuRBO-1: No Improvements','Evaluations'],loc='upper right',fontsize=12) 
        ax[0].tick_params(axis='both', labelsize=16)
        ax[0].set_ylim(4, 10)
        ax[0].grid(True)      
        
        ax[1].plot(output2_minaccum, marker="", lw=3,c='b')
        ax[1].plot(output2, marker=".",linestyle="none",c='b',alpha=0.1)
        ax[1].set_yscale('log')
        ax[1].set_ylabel("NRMSE", fontsize = 16)
        ax[1].set_xlabel("Evaluations", fontsize = 16)
        ax[1].legend(['TuRBO-1: TMP - Best','Evaluations'],loc='upper right',fontsize=12) 
        ax[1].tick_params(axis='both', labelsize=16)
        ax[1].set_ylim(4, 10)
        ax[1].grid(True)

        ax[2].plot(output3_minaccum, marker="", lw=3,c='r')
        ax[2].plot(output3,marker=".",linestyle="none",c='r',alpha=0.1)
        ax[2].set_yscale('log')
        ax[2].set_ylabel("NRMSE", fontsize = 16)
        ax[2].set_xlabel("Evaluations", fontsize = 16)
        ax[2].legend(['TuRBO-1: WTBL & TMP - Best','Evaluations'],loc='upper right',fontsize=12) 
        ax[2].tick_params(axis='both', labelsize=16)
        ax[2].set_ylim(4, 10)
        ax[2].grid(True)
        plt.tight_layout()
        plt.show()         

