# -*- coding: utf-8 -*-
"""
Wrapper for Executing SWAT+ Model

Adapated from Jaya Hafner, Kalcic Lab @ UW Madison

Last updated: 05/07/2025

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
        
        self.param_bsn = [("SURLAG.bsn", 1, 10),
                          ("SFTMP.bsn", -3, 3),
                          ("SMTMP.bsn", -3, 3),
                          ("TIMP.bsn", 0.01, 0.4),
                          ("SMFMX.bsn", 1.4, 6.9),
                          ("SMFMN.bsn", 1.4, 6.9),
                          ("SNOCOVMX.bsn", 0.8, 6),
                          ("SNO50COV.bsn", 0.2, 0.7),
                          ("PHOSKD.bsn", 250, 400),
                          ("PPERCO.bsn", 10, 17.5),
                          ("NPERCO.bsn", 0.01, 1)]
        self.param_hru = [("ESCO.hru", 0.001, 1),
                          ("EPCO.hru", 0, 1),
                          ("R2ADJ.hru", 0.6, 1),
                          ("OV_N.hru", 0.05, 0.5),
                          ("DEP_IMP.hru", 1000, 2500)]
        self.param_mgt = [("CN2.mgt", 74, 91),
                          ("BIOMIX.mgt", 0.15, 0.35)]
        self.param_sdr = [("LATKSATF.sdr", 0.01, 4)]             
        self.param_sol = [("ANION_EXCL.sol", 0.01, 0.5),
                          ("SOL_CRK.sol", 0.05, 0.5),
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
                      
        self.param_list = self.param_bsn + self.param_hru + self.param_mgt + self.param_sdr + self.param_sol
        self.theta_dim = len(self.param_list)
        self.df_param = pd.DataFrame(self.param_list,columns=["parameter","LB","UB"])
        self.LB = torch.tensor(self.df_param.iloc[:,1].tolist())
        self.UB = torch.tensor(self.df_param.iloc[:,2].tolist())
        
        # Specifying ground truth data:
        df1 = pd.read_csv('obtileQ_KD.csv')
        self.dates = pd.date_range(start=f"1/1/2020", periods = 1461).strftime("%m/%d/%Y")
        self.ground_truth = torch.tensor(df1.iloc[:,1:13].to_numpy())
        
                      
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
            theta_str = f'{theta_str:.3f}'
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
            theta_str = f'{theta_str:.3f}'
            new_line_hru[i] = theta_str.rjust(16) + pd.read_csv(self.HRU_iter_path, header=None).loc[i].to_string(index=False)
            old_line_hru[i] = pd.read_csv(self.HRU_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_hru, InputPath_hru)
        with open(InputPath_hru, 'r') as file2:  # Read in the .hru file
            filedata2 = file2.read()     
        
        # Finds old line in .hru file, replaces it with new theta:
        for i in range(len(self.param_hru)):    
            filedata2 = filedata2.replace(old_line_hru[i], new_line_hru[i])
        
        with open(InputPath_hru, 'w') as file2:  # Write the file out again
            file2.write(filedata2)
        file2.close()                

        # MGT file:
        # Create input file with new MGT parameters:    
        mgt_name = '000010001'
        DefaultPath_mgt = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.mgt"
        InputPath_mgt = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + mgt_name + ".mgt"
        
        old_line_mgt = [None]*len(self.param_mgt)
        new_line_mgt = [None]*len(self.param_mgt)
            
        for i in range(len(self.param_mgt)):
            theta_str = theta[i+len(self.param_bsn)+len(self.param_hru)].squeeze(0).tolist()
            theta_str = f'{theta_str:.3f}'
            new_line_mgt[i] = theta_str.rjust(16) + pd.read_csv(self.MGT_iter_path, header=None).loc[i].to_string(index=False)
            old_line_mgt[i] = pd.read_csv(self.MGT_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_mgt, InputPath_mgt)
        with open(InputPath_mgt, 'r') as file3:  # Read in the .mgt file
            filedata3 = file3.read()     
        
        # Finds old line in .mgt file, replaces it with new theta:
        for i in range(len(self.param_mgt)):    
            filedata3 = filedata3.replace(old_line_mgt[i], new_line_mgt[i])
        
        with open(InputPath_mgt, 'w') as file3:  # Write the file out again
            file3.write(filedata3)
        file3.close()  
        
        # SDR file:
        # Create input file with new SDR parameters:    
        sdr_name = '000010001'
        DefaultPath_sdr = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.sdr"
        InputPath_sdr = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + sdr_name + ".sdr"
        
        old_line_sdr = [None]*len(self.param_sdr)
        new_line_sdr = [None]*len(self.param_sdr)
            
        for i in range(len(self.param_sdr)):
            theta_str = theta[i+len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)].squeeze(0).tolist()
            theta_str = f'{theta_str:.3f}'
            new_line_sdr[i] = theta_str.rjust(10) + pd.read_csv(self.SDR_iter_path, header=None).loc[i].to_string(index=False)
            old_line_sdr[i] = pd.read_csv(self.SDR_nom_path, header=None).loc[i].to_string(index=False)
            
        shutil.copy(DefaultPath_sdr, InputPath_sdr)
        with open(InputPath_sdr, 'r') as file4:  # Read in the .sdr file
            filedata4 = file4.read()     
        
        # Finds old line in .sdr file, replaces it with new theta:
        for i in range(len(self.param_sdr)):    
            filedata4 = filedata4.replace(old_line_sdr[i], new_line_sdr[i])
        
        with open(InputPath_sdr, 'w') as file4:  # Write the file out again
            file4.write(filedata4)
        file4.close()  
        
        #######################################################################
        # Next, for the .sol file, we need to treat theta differently, as some 
        # of the theta values occur in the same line.
        #######################################################################
        
        sol_name = '000010001'
        DefaultPath_sol = "C:\\SWAT_Calibration\\Nominal_Input_Files\\000010001.sol"
        InputPath_sol = "C:\\SWAT_Calibration\\Buckeye_TxtInOut\\" + sol_name + ".sol"
        
        
        theta_ANION = theta[1+len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr)].squeeze(0).tolist()
        theta_ANION = f'{theta_ANION:.3f}'
        theta_CRK = theta[2+len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr)].squeeze(0).tolist()
        theta_CRK = f'{theta_CRK:.3f}'
        
        theta_KSAT = [None]*6
        theta_BD = [None]*6
        theta_AWC = [None]*6
        
        for i in range(6):
            theta_KSAT[i] = theta[3+len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr) + i].squeeze(0).tolist()
            theta_KSAT[i] = f'{theta_KSAT[i]:.2f}'
            theta_BD[i] = theta[8+len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr) + i].squeeze(0).tolist()
            theta_BD[i] = f'{theta_BD[i]:.2f}'
            theta_AWC[i] = theta[14+len(self.param_bsn)+len(self.param_hru)+len(self.param_mgt)+len(self.param_sdr)+ i].squeeze(0).tolist()
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
        with open(InputPath_sol, 'r') as file5:  # Read in the .sol file
            filedata5 = file5.read() 
        
        # Find old lines in .sol file, replaces it with new theta:
 
        filedata5 = filedata5.replace(old_line_ANION, new_line_ANION)
        filedata5 = filedata5.replace(old_line_CRK, new_line_CRK)
        filedata5 = filedata5.replace(old_line_KSAT, new_line_KSAT)
        filedata5 = filedata5.replace(old_line_BD, new_line_BD)
        filedata5 = filedata5.replace(old_line_AWC, new_line_AWC)
        
        with open(InputPath_sol, 'w') as file5:  # Write the file out again
            file5.write(filedata5)
        file5.close() 
         
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
        
        # Stacking outputs in order as seen in obtileQ:
            
        sensors = torch.stack([QTILE,SURQ,TILEP,SURP,TILENO3,SURNO3,self.VWC10,self.VWC20,self.VWC50,self.STMP10,self.STMP20,self.STMP50],dim=1)
                 
        return sensors


if __name__== '__main__':
    a = SWATrun()
    
    theta = torch.rand(len(a.param_list))
    
    # Rescaling:
    LB = a.LB
    UB = a.UB
    
    theta_scaled = LB + (UB - LB)*theta
    sensors_alldates = a.model_run(theta_scaled)
    
    













