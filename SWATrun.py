# -*- coding: utf-8 -*-
"""
Wrapper for Executing SWAT+ Model

Adapated from Jaya Hafner, Kalcic Lab @ UW Madison

Last updated: 07/01/2025

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


class SWATrun():
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
        
        self.param_bsn = [("SURLAG.bsn", 0.1, 10),
                          ("SFTMP.bsn", -3, 3),
                          ("SMTMP.bsn", -3, 3),
                          ("TIMP.bsn", 0.01, 0.4),
                          ("SMFMX.bsn", 1.4, 6.9),
                          ("SMFMN.bsn", 1.4, 6.9),
                          ("SNOCOVMX.bsn", 0.8, 6),
                          ("SNO50COV.bsn", 0.2, 0.7),
                          ("PHOSKD.bsn", 100, 400),
                          ("PPERCO.bsn", 10, 17.5),
                          ("NPERCO.bsn", 0.01, 1)]
        self.param_gw = [("GW_DELAY.gw",5,50),
                         ("ALPHA.BF.gw",0.1,1),
                         ("GWQMN.gw",500,1500),
                         ("GW_REVAP.gw",0.02,0.2),
                         ("REVAPMN.gw",600,1000),
                         ("RCHRG_DP.gw",0,1)]
        self.param_hru = [("ESCO.hru", 0.001, 1),
                          ("EPCO.hru", 0, 1),
                          ("R2ADJ.hru", 0, 1),
                          ("OV_N.hru", 0.05, 0.5),
                          ("DEP_IMP.hru", 1000, 2500)]
        self.param_mgt = [("CN2.mgt", 84, 94),
                          ("BIOMIX.mgt", 0.15, 0.35)]
        self.param_sdr = [("LATKSATF.sdr", 1, 4)]             
        self.param_sol = [("ANION_EXCL.sol", 0.01, 0.5),
                          ("SOL_CRK.sol", 0, 0.5),
                          ("KSAT(1).sol", 22, 23),
                          ("KSAT(2).sol", 22, 23),
                          ("KSAT(3).sol", 22, 23),
                          ("KSAT(4).sol", 22, 23),
                          ("KSAT(5).sol", 19, 22),
                          ("KSAT(6).sol", 2, 22),
                          ("SOL_BD(1).sol", 0.9, 1.2),
                          ("SOL_BD(2).sol", 1.1, 1.4),
                          ("SOL_BD(3).sol", 1.2, 1.6),
                          ("SOL_BD(4).sol", 1.2, 1.6),
                          ("SOL_BD(5).sol", 1.4, 2),
                          ("SOL_BD(6).sol", 1.4, 2),
                          ("SOL_AWC(1).sol", 0.2, 0.3),
                          ("SOL_AWC(2).sol", 0.2, 0.4),
                          ("SOL_AWC(3).sol", 0.2, 0.4),
                          ("SOL_AWC(4).sol", 0.1, 0.4),
                          ("SOL_AWC(5).sol", 0.2, 0.4),
                          ("SOL_AWC(6).sol", 0.2, 0.4)]
                      
        self.param_list = self.param_bsn + self.param_gw + self.param_hru + self.param_mgt + self.param_sdr + self.param_sol
        self.theta_dim = len(self.param_list)
        self.df_param = pd.DataFrame(self.param_list,columns=["parameter","LB","UB"])
        self.LB = torch.tensor(self.df_param.iloc[:,1].tolist())
        self.UB = torch.tensor(self.df_param.iloc[:,2].tolist())
        
        # Specifying ground truth data:
        df1 = pd.read_csv('obtileQ_KD.csv')
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
        self.GW_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_GW.txt"
        self.HRU_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_HRU.txt"
        self.MGT_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_MGT.txt"
        self.SDR_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_SDR.txt"
        self.SOL_nom_path = "C:\\SWAT_Calibration\\Nominal_Inputs\\Param_SOL.txt"
        
        # Parameter iteration files to add new thetas to:
        self.BSN_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_BSN.txt"
        self.GW_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_GW.txt"        
        self.HRU_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_HRU.txt"
        self.MGT_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_MGT.txt"
        self.SDR_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_SDR.txt"
        self.SOL_iter_path = "C:\\SWAT_Calibration\\Input_Iterations\\Param_Iter_SOL.txt"            
                
        
    def model_run(self, theta):
        
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
        
        # GW file:
        # Create input file with new BSN parameters:    
        gw_name = '000010001'
        DefaultPath_gw = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.gw"
        InputPath_gw = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + gw_name + ".gw"
        
        old_line_gw = [None]*len(self.param_gw)
        new_line_gw = [None]*len(self.param_gw)
            
        for i in range(len(self.param_gw)):
            theta_str = theta[i + len(self.param_bsn)].squeeze(0).tolist()
            theta_str = f'{theta_str:.5f}'
            new_line_gw[i] = theta_str.rjust(16) + pd.read_csv(self.GW_iter_path, header=None).loc[i].to_string(index=False)
            old_line_gw[i] = pd.read_csv(self.GW_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_gw, InputPath_gw)
        with open(InputPath_gw, 'r') as file2:  # Read in the .bsn file
            filedata2 = file2.read()     
        
        # Finds old line in .bsn file, replaces it with new theta:
        for i in range(len(self.param_gw)):    
            filedata2 = filedata2.replace(old_line_gw[i], new_line_gw[i])
        
        with open(InputPath_gw, 'w') as file2:  # Write the file out again
            file2.write(filedata2)
        file2.close()

        
        # HRU file:
        # Create input file with new HRU parameters:    
        hru_name = '000010001'
        DefaultPath_hru = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.hru"
        InputPath_hru = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + hru_name + ".hru"
        
        old_line_hru = [None]*len(self.param_hru)
        new_line_hru = [None]*len(self.param_hru)
            
        for i in range(len(self.param_hru)):
            theta_str = theta[i+len(self.param_bsn)+len(self.param_gw)].squeeze(0).tolist()
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
            theta_str = theta[i+len(self.param_bsn)+len(self.param_gw)+len(self.param_hru)].squeeze(0).tolist()
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
            theta_str = theta[i+len(self.param_bsn)+len(self.param_gw)+len(self.param_hru)+len(self.param_mgt)].squeeze(0).tolist()
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
        
        
        theta_ANION = theta[len(self.param_bsn)+len(self.param_gw)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr)].squeeze(0).tolist()
        theta_ANION = f'{theta_ANION:.5f}'
        theta_CRK = theta[1+len(self.param_bsn)+len(self.param_gw)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr)].squeeze(0).tolist()
        theta_CRK = f'{theta_CRK:.5f}'
        
        theta_KSAT = [None]*6
        theta_BD = [None]*6
        theta_AWC = [None]*6
        
        for i in range(6):
            theta_KSAT[i] = theta[2+len(self.param_bsn)+len(self.param_gw)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr) + i].squeeze(0).tolist()
            theta_KSAT[i] = f'{theta_KSAT[i]:.2f}'
            theta_BD[i] = theta[8+len(self.param_bsn)+len(self.param_gw)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr) + i].squeeze(0).tolist()
            theta_BD[i] = f'{theta_BD[i]:.2f}'
            theta_AWC[i] = theta[14+len(self.param_bsn)+len(self.param_gw)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr)+ i].squeeze(0).tolist()
            theta_AWC[i] = f'{theta_AWC[i]:.2f}'
        
        new_line_ANION = pd.read_csv(self.SOL_iter_path, header=None).loc[0].to_string(index=False) + theta_ANION
        new_line_CRK = pd.read_csv(self.SOL_iter_path, header=None).loc[1].to_string(index=False) + theta_CRK
        new_line_KSAT = pd.read_csv(self.SOL_iter_path, header=None).loc[2].to_string(index=False) + theta_KSAT[0].rjust(12) + theta_KSAT[1].rjust(12) + theta_KSAT[2].rjust(12) + theta_KSAT[3].rjust(12) + theta_KSAT[4].rjust(12) + theta_KSAT[5].rjust(12)
        new_line_BD = pd.read_csv(self.SOL_iter_path, header=None).loc[3].to_string(index=False) + theta_BD[0].rjust(12) + theta_BD[1].rjust(12) + theta_BD[2].rjust(12) + theta_BD[3].rjust(12) + theta_BD[4].rjust(12) + theta_BD[5].rjust(12)
        new_line_AWC = pd.read_csv(self.SOL_iter_path, header=None).loc[4].to_string(index=False) + theta_AWC[0].rjust(12) + theta_AWC[1].rjust(12) + theta_AWC[2].rjust(12) + theta_AWC[3].rjust(12) + theta_AWC[4].rjust(12) + theta_AWC[5].rjust(12)
        
        old_line_ANION = pd.read_csv(self.SOL_nom_path, header=None).loc[0].to_string(index=False)
        old_line_CRK = pd.read_csv(self.SOL_nom_path, header=None).loc[1].to_string(index=False)
        old_line_KSAT = pd.read_csv(self.SOL_nom_path, header=None).loc[2].to_string(index=False)
        old_line_BD = pd.read_csv(self.SOL_nom_path, header=None).loc[3].to_string(index=False)
        old_line_AWC = pd.read_csv(self.SOL_nom_path, header=None).loc[4].to_string(index=False)
        
        
        shutil.copy(DefaultPath_sol, InputPath_sol)
        with open(InputPath_sol, 'r') as file6:  # Read in the .sol file
            filedata6 = file6.read() 
        
        # Find old lines in .sol file, replaces it with new theta:
 
        filedata6 = filedata6.replace(old_line_ANION, new_line_ANION)
        filedata6 = filedata6.replace(old_line_CRK, new_line_CRK)
        filedata6 = filedata6.replace(old_line_KSAT, new_line_KSAT)
        filedata6 = filedata6.replace(old_line_BD, new_line_BD)
        filedata6 = filedata6.replace(old_line_AWC, new_line_AWC)
        
        with open(InputPath_sol, 'w') as file6:  # Write the file out again
            file6.write(filedata6)
        file6.close() 
         
        #######################################################################
        # Executing SWAT run
        #######################################################################

        start = time.time()
        print('Running SWAT...')
        project_path = "C:\\SWAT_Calibration\\Buckeye_TxtInOut"
        swat_exe = os.path.join(project_path, "r2adj.exe")
        subprocess.run([swat_exe], cwd=project_path)
        end = time.time()
        print('SWAT run complete in' + ' ' + f'{end-start:.4f}' + ' ' + 'seconds.')



        # Obtaining outputs of interest for calibration:
            # For convenience, these are in order of how they appear out of the
            # output.hru file:
        
        self.hru = pd.read_fwf(self.output_hru, skiprows=8)
                
        SURQ = torch.tensor(self.hru.to_numpy()[:,22].astype(float))[1461:2992]
        self.WATTBL = torch.tensor(self.hru.to_numpy()[:,37].astype(float))[1461:2992] # Defining this with self to plot all values later
        QTILE = torch.tensor(self.hru.to_numpy()[:,38].astype(float))[1461:2992]
        self.STMP10 = torch.tensor(self.hru.to_numpy()[:,39].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.STMP20 = torch.tensor(self.hru.to_numpy()[:,40].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.STMP50 = torch.tensor(self.hru.to_numpy()[:,41].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.VWC10 = torch.tensor(self.hru.to_numpy()[:,42].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.VWC20 = torch.tensor(self.hru.to_numpy()[:,43].astype(float))[1461:2992] # Defining this with self to plot all values later
        self.VWC50 = torch.tensor(self.hru.to_numpy()[:,44].astype(float))[1461:2992] # Defining this with self to plot all values later
        TILENO3 = torch.tensor(self.hru.to_numpy()[:,45].astype(float))[1461:2992]
        SURNO3 = torch.tensor(self.hru.to_numpy()[:,46].astype(float))[1461:2992]
        TILEP = torch.tensor(self.hru.to_numpy()[:,47].astype(float))[1461:2992]
        SURP = torch.tensor(self.hru.to_numpy()[:,48].astype(float))[1461:2992]
        
        # Calculating Wilting Point, Field Capacity, and Saturation for appropriate layers:
        # TODO: Include %clay as an adjustable parameter. For now, it is defined using nominal value from .sol file    
        clay = torch.tensor([22,22,30])
        BD = torch.tensor([theta[34].item(),theta[35].item(),theta[36].item()])
        AWC = torch.tensor([theta[40].item(),theta[41].item(),theta[42].item()])
        
        self.wilting = (0.4*clay*BD)/100
        self.field_cap = self.wilting + AWC
        self.sat = torch.ones(3) - (BD/2.65)
        
        # Applying special conditions for layer 5 to Wilting Point and Field Capacity:
        self.field_cap[2] = self.sat[2] - 0.05
        self.wilting[2] = self.field_cap[2] - AWC[2]
               
        # Stacking outputs in order as seen in obtileQ:
            
        sensors = torch.stack([QTILE,SURQ,TILEP,SURP,TILENO3,SURNO3,self.VWC10,self.VWC20,self.VWC50,self.STMP10,self.STMP20,self.STMP50,self.WATTBL],dim=1)
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
        sensors = a.model_run(theta_scaled)
        
    if run_type == ['Input']:
        
        # Rescaling:
        LB = a.LB
        UB = a.UB
        
        theta  = torch.tensor(pd.read_csv('df_X_TuRBO_070225.csv').to_numpy())
        output = torch.tensor(pd.read_csv('df_Y_TuRBO_070225.csv').to_numpy())
        
        theta = theta[torch.argmax(output)]

        theta_scaled = LB + (UB - LB)*theta
        
        sensors = a.model_run(theta_scaled)
        
        # df_theta_TuRBO =  pd.DataFrame(theta_scaled)
        # df_theta_TuRBO.to_csv('df_theta_TuRBO.csv', sep=',', index = False, encoding='utf-8')
        
                               
        
    if plotting == True:    
    
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import datetime  
        
        
        # Calibration Plots
        
        start_tr_date = datetime.date(2020,1,1)
        end_tr_date = datetime.date(2023,12,31)
        all_dates = [start_tr_date + datetime.timedelta(days=i) for i in range((end_tr_date - start_tr_date).days + 1)]
        
        # Sensors 1 & 2:
        fig, ax = plt.subplots(2,1, figsize=(24, 12))

        trts_split = all_dates[int(0.8*a.ground_truth[:,0].shape[0])]       
        ax[0].plot(all_dates, sensors[:,0], color='b', lw=3)
        ax[0].plot(all_dates, a.ground_truth[:,0],marker="x",linestyle="none",c='g')
        ax[0].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[0].set_ylabel("Tile Flow (mm)", fontsize = 16)
        ax[0].set_xlabel("Date", fontsize = 16)
        ax[0].set_xlim(start_tr_date, end_tr_date)
        ax[0].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[0].tick_params(axis='both', labelsize=16)
        ax[0].grid(True)
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,1].shape[0])]
        ax[1].plot(all_dates[0:1096], sensors[:,1][0:1096], color='b', lw=3)
        ax[1].plot(all_dates, a.ground_truth[:,1],marker="x",linestyle="none",c='g')
        ax[1].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[1].set_ylabel("Surface Flow (mm)", fontsize = 16)
        ax[1].set_xlabel("Date", fontsize = 16)
        ax[1].set_xlim(start_tr_date, end_tr_date)
        ax[1].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[1].tick_params(axis='both', labelsize=16)
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Sensors 3 & 4:
        fig, ax = plt.subplots(2,1, figsize=(24, 12))
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,2].shape[0])] 
        ax[0].plot(all_dates, sensors[:,2], color='b', lw=3)
        ax[0].plot(all_dates, a.ground_truth[:,2],marker="x",linestyle="none",c='g')
        ax[0].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[0].set_ylabel("Tile P (g/ha)", fontsize = 16)
        ax[0].set_xlabel("Date", fontsize = 16)
        ax[0].set_xlim(start_tr_date, end_tr_date)
        ax[0].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[0].tick_params(axis='both', labelsize=16)
        ax[0].grid(True)
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,3].shape[0])] 
        ax[1].plot(all_dates, sensors[:,3], color='b', lw=3)
        ax[1].plot(all_dates, a.ground_truth[:,3],marker="x",linestyle="none",c='g')
        ax[1].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[1].set_ylabel("Surface P (g/ha)", fontsize = 16)
        ax[1].set_xlabel("Date", fontsize = 16)
        ax[1].set_xlim(start_tr_date, end_tr_date)
        ax[1].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[1].tick_params(axis='both', labelsize=16)
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.show()
                
        # Sensors 5 & 6:
        fig, ax = plt.subplots(2,1, figsize=(24, 12))
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,4].shape[0])] 
        ax[0].plot(all_dates, sensors[:,4], color='b', lw=3)
        ax[0].plot(all_dates, a.ground_truth[:,4],marker="x",linestyle="none",c='g')
        ax[0].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[0].set_ylabel("Tile NO3 (kg/ha)", fontsize = 16)
        ax[0].set_xlabel("Date", fontsize = 16)
        ax[0].set_xlim(start_tr_date, end_tr_date)
        ax[0].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[0].tick_params(axis='both', labelsize=16)
        ax[0].grid(True)
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,5].shape[0])] 
        ax[1].plot(all_dates, sensors[:,5], color='b', lw=3)
        ax[1].plot(all_dates, a.ground_truth[:,5],marker="x",linestyle="none",c='g')
        ax[1].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[1].set_ylabel("Surface NO3 (kg/ha)", fontsize = 16)
        ax[1].set_xlabel("Date", fontsize = 16)
        ax[1].set_xlim(start_tr_date, end_tr_date)
        ax[1].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[1].tick_params(axis='both', labelsize=16)
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Sensors 7,8,9:
        fig, ax = plt.subplots(3,1, figsize=(24, 12))
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,6].shape[0])]
        ax[0].plot(all_dates, sensors[:,6], color='b', lw=3)
        ax[0].plot(all_dates, a.ground_truth[:,6],marker="x",linestyle="none",c='g')
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
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,7].shape[0])]
        ax[1].plot(all_dates, sensors[:,7], color='b', lw=3)
        ax[1].plot(all_dates, a.ground_truth[:,7],marker="x",linestyle="none",c='g')
        ax[1].axhline(y=a.wilting[1],color='red',xmin=0,xmax=1461)
        ax[1].axhline(y=a.field_cap[1],color='black',xmin=0,xmax=1461)
        ax[1].axhline(y=a.sat[1],color='green',xmin=0,xmax=1461)
        ax[1].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[1].set_ylabel("VWC - 20 cm", fontsize = 16)
        ax[1].set_xlabel("Date", fontsize = 16)
        ax[1].set_xlim(start_tr_date, end_tr_date)
        # ax[1].legend(['TuRBO','Ground Truth'],loc='upper center',fontsize=16, ncol=2) 
        ax[1].tick_params(axis='both', labelsize=16)
        ax[1].grid(True)
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,8].shape[0])]
        ax[2].plot(all_dates, sensors[:,8], color='b', lw=3)
        ax[2].plot(all_dates, a.ground_truth[:,8],marker="x",linestyle="none",c='g')
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
        line1, = ax[0].plot([], [], color='b', lw=3)              # TuRBO
        line2, = ax[0].plot([], [], marker="x", linestyle="none", color='g')  # Ground Truth
        line3 = plt.Line2D([], [], color='red')             # Wilting Point
        line4 = plt.Line2D([], [], color='black')           # Field Capacity
        line5 = plt.Line2D([], [], color='green')           # Saturation
        line6 = plt.Line2D([], [], color='black', linestyle='--')
        
        fig.legend([line1, line2, line3, line4, line5, line6],
                   ['TuRBO', 'Ground Truth', 'Wilting Point', 'Field Capacity', 'Saturation','Train/Test Split'],
                   loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        
        # Sensors 10,11,12:
        fig, ax = plt.subplots(3,1, figsize=(24, 12))
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,9].shape[0])]
        ax[0].plot(all_dates, sensors[:,9], color='b', lw=3)
        ax[0].plot(all_dates, a.ground_truth[:,9],marker="x",linestyle="none",c='g')
        ax[0].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[0].set_ylabel("TMP - 10 cm", fontsize = 16)
        ax[0].set_xlabel("Date", fontsize = 16)
        ax[0].set_xlim(start_tr_date, end_tr_date)
        ax[0].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[0].tick_params(axis='both', labelsize=16)
        ax[0].grid(True)
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,10].shape[0])]
        ax[1].plot(all_dates, sensors[:,10], color='b', lw=3)
        ax[1].plot(all_dates, a.ground_truth[:,10],marker="x",linestyle="none",c='g')
        ax[1].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[1].set_ylabel("TMP - 20 cm", fontsize = 16)
        ax[1].set_xlabel("Date", fontsize = 16)
        ax[1].set_xlim(start_tr_date, end_tr_date)
        ax[1].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[1].tick_params(axis='both', labelsize=16)
        ax[1].grid(True)
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,11].shape[0])]
        ax[2].plot(all_dates, sensors[:,11], color='b', lw=3)
        ax[2].plot(all_dates, a.ground_truth[:,11],marker="x",linestyle="none",c='g')
        ax[2].axvline(x=trts_split, color='black', linestyle='--', lw=3)
        ax[2].set_ylabel("TMP - 50 cm", fontsize = 16)
        ax[2].set_xlabel("Date", fontsize = 16)
        ax[2].set_xlim(start_tr_date, end_tr_date)
        ax[2].legend(['TuRBO','Ground Truth','Train/Test Split'],loc='upper center',fontsize=16, ncol=3) 
        ax[2].tick_params(axis='both', labelsize=16)
        ax[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
        # Sensor 13:
            
        fig, ax = plt.subplots(figsize=(24, 6))
        
        trts_split = all_dates[int(0.8*a.ground_truth[:,12].shape[0])+160]
        plt.plot(all_dates[731:1461], sensors[:,12][731:1461], color='b', lw=3)  
        plt.plot(all_dates[731:1461], a.ground_truth[:,12][731:1461],marker="x",linestyle="none",c='g')
        plt.axvline(x=trts_split, color='black', linestyle='--', lw=3)
        plt.ylabel("Water Table Distance (mm)", fontsize = 16)
        plt.xlabel("Date", fontsize = 16)
        plt.legend(['TuRBO','Ground Truth','Train/Test Split'],loc='lower center',fontsize=16,ncol=3) 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([0,1800])
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()












