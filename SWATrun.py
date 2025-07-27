# -*- coding: utf-8 -*-
"""
Wrapper for Executing SWAT+ Model

Adapated from Jaya Hafner, Kalcic Lab @ UW Madison

Last updated: 07/21/2025

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
from datetime import datetime, timedelta
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time


class SWATrun:
    def __init__(self):
        """
        Defines parameter values and ranges for Single-Field SWAT Model.
        -----------------------------------------------------------------------
        param_list: List where oarameter names and bounds for SWAT model are specified.
            -> Parameter name should also have appropriate SWAT file extension.
        df_param: Dataframe of Param_list items.
        LB: Tensor of lower bounds for parameters.
        UB: Tensor of upper bounds for parameters.
             
        """
        
        self.param_bsn = [("SFTMP.bsn", -3, 3),
                          ("SMTMP.bsn", -3, 3),
                          ("TIMP.bsn", 0.01, 0.4),
                          ("SMFMX.bsn", 1.4, 6.9),
                          ("SMFMN.bsn", 1.4, 6.9),
                          ("SNOCOVMX.bsn", 0.8, 6),
                          ("SNO50COV.bsn", 0.2, 0.7)]
        self.param_hru = [("ESCO.hru", 0.001, 1),
                          ("EPCO.hru", 0, 1),
                          ("R2ADJ.hru", 0, 1),
                          ("OV_N.hru", 0.05, 0.5),
                          ("DEP_IMP.hru", 1524, 2500)]
        self.param_mgt = [("CN2.mgt", 84, 94)]
        self.param_sdr = [("LATKSATF.sdr", 1, 4)]             
        self.param_sol = [("SOL_CRK.sol", 0, 0.5)]
                      
        self.param_list = self.param_bsn + self.param_hru + self.param_mgt + self.param_sdr + self.param_sol
        self.theta_dim = len(self.param_list)
        self.df_param = pd.DataFrame(self.param_list,columns=["parameter","LB","UB"])
        self.LB = torch.tensor(self.df_param.iloc[:,1].tolist())
        self.UB = torch.tensor(self.df_param.iloc[:,2].tolist())
        
        # Specifying ground truth data:
        df1 = pd.read_csv('obtileQmin.csv')
        self.dates = pd.date_range(start=f"1/1/2020", periods = 1461).strftime("%m/%d/%Y")
        self.ground_truth = torch.tensor(df1.iloc[:,1:14].to_numpy())
        
                      
        # Define paths to input, output, and executable:
        
        self.output_hru = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\output.hru" 
        
        # Set simulation start and end dates
        self.output_start_date = datetime(2016, 1, 1)
        self.output_end_date = datetime(2023, 12, 31)

        # Generate all days in the range
        self.days = [
            (self.output_start_date + timedelta(days=i)).timetuple().tm_yday
            for i in range((self.output_end_date - self.output_start_date).days + 1)
        ]
        
        # Nominal parameters path to search inputs files for:
        self.BSN_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_BSN.txt"
        self.HRU_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_HRU.txt"
        self.MGT_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_MGT.txt"
        self.SDR_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_SDR.txt"
        self.SOL_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_SOL.txt"
        
        # Parameter iteration files to add new thetas to:
        self.BSN_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_BSN.txt"       
        self.HRU_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_HRU.txt"
        self.MGT_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_MGT.txt"
        self.SDR_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_SDR.txt"
        self.SOL_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_SOL.txt"            
                
        
    def __call__(self, theta):
        
        pd.set_option('display.max_colwidth', None)
        
        # BSN file:
        # Create input file with new BSN parameters:    
        bsn_name = 'basins'
        DefaultPath_bsn = "C:\\SWAT_Calibration\\Nominal_Input_Files\\basins.bsn"
        InputPath_bsn = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + bsn_name + ".bsn"
        
        old_line_bsn = [None]*len(self.param_bsn)
        new_line_bsn = [None]*len(self.param_bsn)
            
        for i in range(len(self.param_bsn)):
            theta_str = theta[i].squeeze(0).tolist()
            theta_str = f'{theta_str:.5f}'
            new_line_bsn[i] = theta_str.rjust(16) + pd.read_csv(self.BSN_iter_path, header=None).loc[i].to_string(index=False)
            old_line_bsn[i] = pd.read_csv(self.BSN_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_bsn, InputPath_bsn)
        with open(InputPath_bsn, 'r') as file1:  # Read in the .bsn file
            filedata1 = file1.read()     
        
        # Finds old line in .bsn file, replaces it with new theta:
        for i in range(len(self.param_bsn)):    
            filedata1 = filedata1.replace(old_line_bsn[i], new_line_bsn[i])
        
        with open(InputPath_bsn, 'w') as file1:  # Write the file out again
            file1.write(filedata1)
        file1.close()
        
        # HRU file:
        # Create input file with new HRU parameters:    
        hru_name = '000010001'
        DefaultPath_hru = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.hru"
        InputPath_hru = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + hru_name + ".hru"
        
        old_line_hru = [None]*len(self.param_hru)
        new_line_hru = [None]*len(self.param_hru)
            
        for i in range(len(self.param_hru)):
            theta_str = theta[i+len(self.param_bsn)].squeeze(0).tolist()
            theta_str = f'{theta_str:.5f}'
            new_line_hru[i] = theta_str.rjust(16) + pd.read_csv(self.HRU_iter_path, header=None).loc[i].to_string(index=False)
            old_line_hru[i] = pd.read_csv(self.HRU_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_hru, InputPath_hru)
        with open(InputPath_hru, 'r') as file3:  # Read in the .hru file
            filedata3 = file3.read()     
        
        # Finds old line in .hru file, replaces it with new theta:
        for i in range(len(self.param_hru)):    
            filedata3 = filedata3.replace(old_line_hru[i], new_line_hru[i])
        
        with open(InputPath_hru, 'w') as file3:  # Write the file out again
            file3.write(filedata3)
        file3.close()                

        # MGT file:
        # Create input file with new MGT parameters:    
        mgt_name = '000010001'
        DefaultPath_mgt = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.mgt"
        InputPath_mgt = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + mgt_name + ".mgt"
        
        old_line_mgt = [None]*len(self.param_mgt)
        new_line_mgt = [None]*len(self.param_mgt)
            
        for i in range(len(self.param_mgt)):
            theta_str = theta[i+len(self.param_bsn)+len(self.param_hru)].squeeze(0).tolist()
            theta_str = f'{theta_str:.5f}'
            new_line_mgt[i] = theta_str.rjust(16) + pd.read_csv(self.MGT_iter_path, header=None).loc[i].to_string(index=False)
            old_line_mgt[i] = pd.read_csv(self.MGT_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_mgt, InputPath_mgt)
        with open(InputPath_mgt, 'r') as file4:  # Read in the .mgt file
            filedata4 = file4.read()     
        
        # Finds old line in .mgt file, replaces it with new theta:
        for i in range(len(self.param_mgt)):    
            filedata4 = filedata4.replace(old_line_mgt[i], new_line_mgt[i])
        
        with open(InputPath_mgt, 'w') as file4:  # Write the file out again
            file4.write(filedata4)
        file4.close()  
        
        # SDR file:
        # Create input file with new SDR parameters:    
        sdr_name = '000010001'
        DefaultPath_sdr = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.sdr"
        InputPath_sdr = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + sdr_name + ".sdr"
        
        old_line_sdr = [None]*len(self.param_sdr)
        new_line_sdr = [None]*len(self.param_sdr)
            
        for i in range(len(self.param_sdr)):
            theta_str = theta[i+len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)].squeeze(0).tolist()
            theta_str = f'{theta_str:.5f}'
            new_line_sdr[i] = theta_str.rjust(10) + pd.read_csv(self.SDR_iter_path, header=None).loc[i].to_string(index=False)
            old_line_sdr[i] = pd.read_csv(self.SDR_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_sdr, InputPath_sdr)
        with open(InputPath_sdr, 'r') as file5:  # Read in the .sdr file
            filedata5 = file5.read()     
        
        # Finds old line in .sdr file, replaces it with new theta:
        for i in range(len(self.param_sdr)):    
            filedata5 = filedata5.replace(old_line_sdr[i], new_line_sdr[i])
        
        with open(InputPath_sdr, 'w') as file5:  # Write the file out again
            file5.write(filedata5)
        file5.close()  
        
        #######################################################################
        # Next, for the .sol file, we need to treat theta differently, as some 
        # of the theta values occur in the same line.
        #######################################################################
        
        sol_name = '000010001'
        DefaultPath_sol = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.sol"
        InputPath_sol = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + sol_name + ".sol"
        
        theta_CRK = theta[len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr)].squeeze(0).tolist()
        theta_CRK = f'{theta_CRK:.5f}'
        
        
        new_line_CRK = pd.read_csv(self.SOL_iter_path, header=None).loc[0].to_string(index=False) + theta_CRK
        
        old_line_CRK = pd.read_csv(self.SOL_nom_path, header=None).loc[0].to_string(index=False)

        
        shutil.copy(DefaultPath_sol, InputPath_sol)
        with open(InputPath_sol, 'r') as file6:  # Read in the .sol file
            filedata6 = file6.read() 
        
        # Find old lines in .sol file, replaces it with new theta:
        filedata6 = filedata6.replace(old_line_CRK, new_line_CRK)

        with open(InputPath_sol, 'w') as file6:  # Write the file out again
            file6.write(filedata6)
        file6.close() 
         
        #######################################################################
        # Executing SWAT run
        #######################################################################

        start = time.time()
        print('Running SWAT...')
        project_path = "C:\\SWAT_Calibration\\Buckeye_TxtInOut"
        swat_exe = os.path.join(project_path, "depimpchange.exe")
        subprocess.run([swat_exe], cwd=project_path)
        end = time.time()
        print('SWAT run complete in' + ' ' + f'{end-start:.4f}' + ' ' + 'seconds.')



        # Obtaining outputs of interest for calibration:
            # For convenience, these are in order of how they appear out of the
            # output.hru file:
        
        self.hru = pd.read_fwf(self.output_hru, skiprows=8)
                
        #SURQ = torch.tensor(self.hru.to_numpy()[:,22].astype(float))[1461:2992]
        self.WATTBL = torch.tensor(self.hru.to_numpy()[:,37].astype(float))[1461:2992] # Defining this with self to plot all values later
        QTILE = torch.tensor(self.hru.to_numpy()[:,38].astype(float))[1461:2992]
        self.STMP10 = torch.tensor(self.hru.to_numpy()[:,39].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.STMP20 = torch.tensor(self.hru.to_numpy()[:,40].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.STMP50 = torch.tensor(self.hru.to_numpy()[:,41].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.VWC10 = torch.tensor(self.hru.to_numpy()[:,42].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.VWC20 = torch.tensor(self.hru.to_numpy()[:,43].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.VWC50 = torch.tensor(self.hru.to_numpy()[:,44].astype(float))[1461:2992] # Defining this with self to plot all values later
        # TILENO3 = torch.tensor(self.hru.to_numpy()[:,45].astype(float))[1461:2992]
        # SURNO3 = torch.tensor(self.hru.to_numpy()[:,46].astype(float))[1461:2992]
        # TILEP = torch.tensor(self.hru.to_numpy()[:,47].astype(float))[1461:2992]
        # SURP = torch.tensor(self.hru.to_numpy()[:,48].astype(float))[1461:2992]
        
        # Calculating Wilting Point, Field Capacity, and Saturation for appropriate layers:
        # TODO: Include %clay as an adjustable parameter. For now, it is defined using nominal value from .sol file    
        
        clay = torch.tensor([22,22,42])
        BD = torch.tensor([1.11,1.25,1.26])
        AWC = torch.tensor([0.24,0.32,0.23])
        
        self.wilting = (0.4*clay*BD)/100
        self.field_cap = self.wilting + AWC
        self.sat = torch.ones(3) - (BD/2.65)
                      
        # Stacking outputs in order as seen in obtileQ:
        
        sensors = torch.stack([QTILE,self.VWC10,self.VWC50,self.STMP10,self.STMP50,self.WATTBL],dim=1)    
        
        # sensors = torch.stack([QTILE,SURQ,TILEP,SURP,TILENO3,SURNO3,self.VWC10,self.VWC20,self.VWC50,self.STMP10,self.STMP20,self.STMP50,self.WATTBL],dim=1)
        # sensors = torch.stack([QTILE,SURQ,TILEP,SURP,self.VWC10,self.VWC20,self.VWC50,self.STMP10,self.STMP20,self.STMP50,self.WATTBL],dim=1)    

             
        return sensors


if __name__== '__main__':
    a = SWATrun()
    dim = a.theta_dim
    run_type = ['Input'] # Types accepted: ['Rand','Input']
    
    plotting = True # Option for turning plotting on/off
    
    if run_type == ['Rand']:
        theta = torch.rand(dim)
    
        # Rescaling:
        LB = a.LB
        UB = a.UB
    
        theta_scaled = LB + (UB - LB)*theta
        sensors = a(theta_scaled)
        
    if run_type == ['Input']:
        
        # Rescaling:
        LB = a.LB
        UB = a.UB
        
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
               
        # # sensors1 = a(theta1_best)
        # # sensors2 = a(theta2_best)
        sensors3 = a(theta3_best)
        
        # df_theta3_sensors =  pd.DataFrame(sensors3)
        # df_theta3_sensors.to_csv('df_theta3_sensors.csv', sep=',', index = False, encoding='utf-8')
        
        # # sensors1_VWC20 = a.VWC20
        # # sensors2_VWC20 = a.VWC20
        # sensors3_VWC20 = a.VWC20
        
        sensors1 = torch.tensor(pd.read_csv('df_theta1_sensors.csv').to_numpy())
        sensors2 = torch.tensor(pd.read_csv('df_theta2_sensors.csv').to_numpy())
        sensors3 = torch.tensor(pd.read_csv('df_theta3_sensors.csv').to_numpy())
                 
        sensors1_VWC20 = torch.tensor(pd.read_csv('sensor1_VWC20.csv').to_numpy())
        sensors2_VWC20 = torch.tensor(pd.read_csv('sensor2_VWC20.csv').to_numpy())
        sensors3_VWC20 = torch.tensor(pd.read_csv('sensor3_VWC20.csv').to_numpy())
                
    if plotting == True:    
    
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import datetime  
        
        
        # Calibration Plots
        
        start_tr_date = datetime.date(2020,1,1)
        end_tr_date = datetime.date(2023,12,31)
        all_dates = [start_tr_date + datetime.timedelta(days=i) for i in range((end_tr_date - start_tr_date).days + 1)]
        
        # # Sensors 1 & 2:

        fig, ax = plt.subplots(figsize=(24,6))  
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,0].shape[0])]
        plt.plot(all_dates, sensors1[:,0], color='orange', lw=3)
        plt.plot(all_dates, sensors2[:,0], color='b', lw=3)
        plt.plot(all_dates, sensors3[:,0], color='r', lw=3,linestyle='--')
        plt.plot(all_dates, a.ground_truth[:,0],marker="x",linestyle="none",c='g')
        plt.axvline(x=trts_split, color='black', linestyle='--', lw=3)
        plt.ylabel("Tile Flow (mm)", fontsize = 16)
        plt.xlabel("Date", fontsize = 16)
        plt.legend(['TuRBO-1: No Improvements','TuRBO-1: TMP','TuRBO-1: WTBL & TMP','Ground Truth'],loc='upper right',fontsize=12) 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
                
        # Sensors 7,8,9:
        fig, ax = plt.subplots(3,1, figsize=(24, 12))
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,2].shape[0])]
        ax[0].plot(all_dates, sensors1[:,1], color='orange', lw=3)
        ax[0].plot(all_dates, sensors2[:,1], color='b', lw=3)
        ax[0].plot(all_dates, sensors3[:,1], color='r', lw=3)
        ax[0].plot(all_dates, a.ground_truth[:,1],marker="x",linestyle="none",c='g')
        ax[0].axhline(y=a.wilting[0],color='red',xmin=0,xmax=1461)
        ax[0].axhline(y=a.field_cap[0],color='black',xmin=0,xmax=1461)
        ax[0].axhline(y=a.sat[0],color='green',xmin=0,xmax=1461)
        ax[0].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[0].set_ylabel("VWC - 10 cm", fontsize = 16)
        ax[0].set_xlabel("Date", fontsize = 16)
        ax[0].set_xlim(start_tr_date, end_tr_date)
        # ax[0].legend(['TuRBO','Ground Truth'],loc='upper center',fontsize=16, ncol=2) 
        ax[0].tick_params(axis='both', labelsize=16)
        ax[0].grid(True)
        
        # VWC 20 CM FOR VALIDATION
        ax[1].plot(all_dates, sensors1_VWC20, color='orange', lw=3)
        ax[1].plot(all_dates, sensors2_VWC20, color='b', lw=3)
        ax[1].plot(all_dates, sensors3_VWC20, color='r', lw=3)
        ax[1].plot(all_dates, a.ground_truth[:,10],marker="x",linestyle="none",c='g')
        ax[1].axhline(y=a.wilting[1],color='red',xmin=0,xmax=1461)
        ax[1].axhline(y=a.field_cap[1],color='black',xmin=0,xmax=1461)
        ax[1].axhline(y=a.sat[1],color='green',xmin=0,xmax=1461)
        ax[1].set_ylabel("VWC - 20 cm", fontsize = 16)
        ax[1].set_xlabel("Date", fontsize = 16)
        ax[1].set_xlim(start_tr_date, end_tr_date)
        # ax[1].legend(['TuRBO','Ground Truth'],loc='upper center',fontsize=16, ncol=2) 
        ax[1].tick_params(axis='both', labelsize=16)
        ax[1].grid(True)
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,3].shape[0])]
        ax[2].plot(all_dates, sensors1[:,2], color='orange', lw=3)
        ax[2].plot(all_dates, sensors2[:,2], color='b', lw=3)
        ax[2].plot(all_dates, sensors3[:,2], color='r', lw=3)
        ax[2].plot(all_dates, a.ground_truth[:,2],marker="x",linestyle="none",c='g')
        ax[2].axhline(y=a.wilting[2],color='red',xmin=0,xmax=1461)
        ax[2].axhline(y=a.field_cap[2],color='black',xmin=0,xmax=1461)
        ax[2].axhline(y=a.sat[2],color='green',xmin=0,xmax=1461)
        ax[2].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[2].set_ylabel("VWC - 50 cm", fontsize = 16)
        ax[2].set_xlabel("Date", fontsize = 16)
        ax[2].set_xlim(start_tr_date, end_tr_date)
        # ax[2].legend(['TuRBO','Ground Truth'],loc='upper center',fontsize=16, ncol=2) 
        ax[2].tick_params(axis='both', labelsize=16)
        ax[2].grid(True)
        
        handles, labels = ax[0].get_legend_handles_labels()
        line1, = ax[0].plot([], [], color='orange', lw=3)              # TuRBO
        line2, = ax[0].plot([], [], color='b', lw=3)              # TuRBO
        line3, = ax[0].plot([], [], color='r', lw=3)              # TuRBO
        line4, = ax[0].plot([], [], marker="x", linestyle="none", color='g')  # Ground Truth
        line5 = plt.Line2D([], [], color='red')             # Wilting Point
        line6 = plt.Line2D([], [], color='black')           # Field Capacity
        line7 = plt.Line2D([], [], color='green')           # Saturation
        line8 = plt.Line2D([], [], color='black', linestyle='--')
        
        fig.legend([line1, line2, line3, line4, line5, line6, line7, line8],
                    ['TuRBO-1: No Improvement','TuRBO-1: TMP','TuRBO-1: TMP & WTBL', 'Ground Truth', 'Wilting Point', 'Field Capacity', 'Saturation','Train/Test Split'],
                    loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        
        # # Sensors 10,11,12:
        # fig, ax = plt.subplots(3,1, figsize=(24, 12))
        
        # trts_split = all_dates[int(0.8*a.ground_truth[:,4].shape[0])]
        # ax[0].plot(all_dates, sensors3[:,3], color='r', lw=3)
        # ax[0].plot(all_dates, a.ground_truth[:,3],marker="x",linestyle="none",c='g')
        # ax[0].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        # ax[0].set_ylabel("TMP - 10 cm", fontsize = 16)
        # ax[0].set_xlabel("Date", fontsize = 16)
        # ax[0].set_xlim(start_tr_date, end_tr_date)
        # ax[0].set_ylim([-10,35])
        # ax[0].legend(['TuRBO-1: TMP & WTBL','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        # ax[0].tick_params(axis='both', labelsize=16)
        # ax[0].grid(True)
        
        # # TMP 20 CM FOR VALIDATION
        # trts_split = all_dates[int(0.8*a.ground_truth[:,11].shape[0])]
        # ax[1].plot(all_dates, a.STMP20, color='r', lw=3)
        # ax[1].plot(all_dates, a.ground_truth[:,11],marker="x",linestyle="none",c='g')
        # ax[1].set_ylabel("TMP - 20 cm", fontsize = 16)
        # ax[1].set_xlabel("Date", fontsize = 16)
        # ax[1].set_xlim(start_tr_date, end_tr_date)
        # ax[1].set_ylim([-5,35])
        # ax[1].legend(['TuRBO-1: TMP & WTBL','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        # ax[1].tick_params(axis='both', labelsize=16)
        # ax[1].grid(True)
        
        # trts_split = all_dates[int(0.8*a.ground_truth[:,5].shape[0])]
        # ax[2].plot(all_dates, sensors3[:,4], color='r', lw=3)
        # ax[2].plot(all_dates, a.ground_truth[:,4],marker="x",linestyle="none",c='g')
        # ax[2].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        # ax[2].set_ylabel("TMP - 50 cm", fontsize = 16)
        # ax[2].set_xlabel("Date", fontsize = 16)
        # ax[2].set_xlim(start_tr_date, end_tr_date)
        # ax[2].set_ylim([-5,35])
        # ax[2].legend(['TuRBO-1: TMP & WTBL','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        # ax[2].tick_params(axis='both', labelsize=16)
        # ax[2].grid(True)
        
        # plt.tight_layout()
        # plt.show()
    
        # Sensor 13:
        
        precip = "C:\\SWAT_Calibration\\PrecipData.txt"
        precip = pd.read_fwf(precip).to_numpy()
        
        fig, ax1 = plt.subplots(figsize=(24, 6))
        ax2 = ax1.twinx()
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,5].shape[0])+50]
        ax1.plot(all_dates[731:1461], sensors1[:,5][731:1461], color='orange', lw=3)
        ax1.plot(all_dates[731:1461], sensors2[:,5][731:1461], color='b', lw=3)
        ax1.plot(all_dates[731:1461], sensors3[:,5][731:1461], color='r', lw=3,linestyle="--")
        ax1.plot(all_dates[731:1461], a.ground_truth[:,5][731:1461],marker="x",linestyle="none",c='g')
        ax1.axvline(x=trts_split, color='black', linestyle='--', lw=3)
        
        ax2.plot(all_dates[731:1461],precip[730:1461], color='k',lw=2)
        
        ax2.set_ylabel("Precipitation (mm)", fontsize = 16)
        ax1.set_ylabel("Water Table Distance (mm)", fontsize = 16)
        ax1.set_xlabel("Date", fontsize = 16)
        ax1.tick_params(axis='both', labelsize=16)
        ax1.legend(['TuRBO-1: No Improvement','TuRBO-1: TMP','TuRBO-1: WTBL & TMP','Ground Truth','Train/Test Split'],loc='upper right',fontsize=16,ncol=1) 
        ax2.legend(['Prec.'],loc='lower left',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([0,1800])
        ax1.invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()












