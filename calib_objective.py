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

import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.double)

seed = 0
simulator = SWATrun()
input_dim = simulator.theta_dim

    
def objective_function(theta,method = ['NRMSE']):  
    
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
       
    if method == ['NRMSE']:
        
        train_split = 0.8 # Use this for treating 80% of the output for testing
        
        output = torch.zeros(sensors.size(1))
        for i in range(sensors.size(1)):
            output[i] = torch.sqrt(torch.sum(torch.square((sensors[:,i][mask[:,i]]-ground_truth[:,i][mask[:,i]])/len(ground_truth[:,i][mask[:,i]])))/ground_truth[:,i][mask[:,i]].max())
         
    else:
        print('Please enter an accepted objective function.')
        output = 0
    
    return output

if __name__== '__main__':
    theta = torch.rand(input_dim)
    output = objective_function(theta,['NRMSE'])
             

  

                  

